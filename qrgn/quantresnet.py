import torch
import numpy as np
from torch import nn
from typing import Optional
from torch import Tensor
from torch_geometric.nn import MessagePassing

class QuantizedActivation(torch.nn.Module):
    def __init__(self, num_sub_kernels: int = 1, activation: str = "softmax", tau: int = 0.1):
        super(QuantizedActivation, self).__init__()
        assert num_sub_kernels >= 1, "num kernels used in convolution (num_sub_kernels) cannot be less than 1"
        self.gumbel_tau = tau
        self.activation = activation
        self.num_sub_kernels = num_sub_kernels

        # Selectively define the activations
        if activation == "sigmoid":
            self.sigmoid = torch.nn.Sigmoid()
        elif activation == "softmax":
            self.softmax = torch.nn.Softmax(dim=-1)
        elif activation == "gumbel_softmax":
            self.gumbel_softmax = self.custom_gumbel_softmax
    
    # Pickling requires actual function names 
    def custom_gumbel_softmax(self, x: Tensor):
        return torch.nn.functional.gumbel_softmax(x, tau=self.gumbel_tau, dim=-1)
    
    def forward(self, x: Tensor):
        """
        x: [num_nodes, node_features] 
           This is the output (logits) 
           which we feed through this custom 
           activation layer
        """
        # Apply the softmax or sigmoid function to generate outputs
        if self.activation == "sigmoid":
            out = torch.round((self.num_sub_kernels - 1) * self.sigmoid(x.squeeze(dim=-1).mean(dim=-1)))
        if self.activation == "softmax":
            out = torch.argmax(self.softmax(x.squeeze(dim=-1)).squeeze(dim=0), dim=-1)
        elif self.activation == "gumbel_softmax":
            out = torch.argmax(self.gumbel_softmax(x.squeeze(dim=-1)).squeeze(dim=0), dim=-1)

        # Return output
        return out


class QuantResNet(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_sub_kernels = 3,
                 use_shared_bias = True,
                 edge_attr_dim = -1, 
                 pos_descr_dim = -1,
                 quant_net_depth = 2,
                 quant_net_expansion = 2,
                 apply_mixture_net = True, 
                 mixture_net_depth = 2,
                 mixture_net_expansion = 2,
                 aggr="add", 
                 device="cpu"):
        """
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        edge_attr_dim - dimension of edge attribute features
        num_sub_kernels: This is the number of subkernels we pre-define for the convolution
                     These num kernels are simply centroids within neighborhoods mapped
                     out onto a 2D cartesian plane. Nodes are mapped to these fixed 
                     centroids (each with their own kernels) and are convolved.
        use_shared_bias: (bool) whether to constrain all subkernels to use a single bias term
                         as is in the CNN convolution kernel definition
        pos_descr_dim: dimension of the positional descriptors to be used
        quant_net_depth: depth of the MLP that learns the neighborhood-kernel quantization
        quant_net_expansion: defines expansion factor for the high dimensional space in which
                             to learn relative feature distances
        apply_mixture_net: (bool) whether or not to combine convolution-like messages post-aggregation
                           with old messages from all nodes in the distinct manner indicated in the paper 
        mixture_net_depth: the depth of the network combining features from different learning paths
        mixture_net_expansion: the expansion (& simultaneously shrinkage factor) for the mixture net
        aggr: how to aggregate node messages
        device: where model lives/runs e.g., cpu or cuda* (gpu)
        """
        super(QuantResNet, self).__init__(aggr=aggr)

        # Assert that model initialization is correct
        # Assert that num_sub_kernels override is expected for this model
        assert not (num_sub_kernels <= 0), "num_sub_kernels, if specified, must be >= 1"
        if pos_descr_dim != -1: assert not (pos_descr_dim <= 0), "positional descriptors must be >= 1"

        # save reference to channels params ...
        self.aggr = aggr
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        self.pos_descr_dim = pos_descr_dim
        self.use_shared_bias = use_shared_bias
        self.num_sub_kernels = num_sub_kernels
        self.pos_descr_exists = pos_descr_dim != -1
        self.edge_attr_exists = edge_attr_dim != -1

        # Mixture MLP Depth
        self.apply_mixture_net = apply_mixture_net
        self.mixture_net_depth = mixture_net_depth
        self.mixture_net_expansion = mixture_net_expansion

        # Quantizer Net Params
        self.quant_net_depth = quant_net_depth
        self.quant_net_expansion = quant_net_expansion

        # Define norm. and activation for update() stage of message passing
        self.relu = torch.nn.ReLU()
        self.bn1d = torch.nn.BatchNorm1d(self.out_channels).to(self.device)
        
        # Define subkernels for convolution-like operation in node neighborhoods
        self.nodes_sub_kernels = self.init_sub_kernels(self.in_channels, self.out_channels, self.num_sub_kernels, use_bias=(not self.use_shared_bias))
        if self.use_shared_bias: self.nodes_shared_bias = torch.nn.Parameter(torch.rand(1, self.out_channels), requires_grad=True).to(self.device)

        # Define subkernels for convolution-like operation on edge_attr if they exist
        if self.edge_attr_exists:
            assert isinstance(edge_attr_dim, int), "Num. Edge Attributes/Features (edge_attr_dim) is expected to be an int"
            self.edges_sub_kernels = self.init_sub_kernels(self.edge_attr_dim, self.out_channels, self.num_sub_kernels, use_bias=(not self.use_shared_bias))
            if self.use_shared_bias: self.edges_shared_bias = torch.nn.Parameter(torch.rand(1, self.out_channels), requires_grad=True).to(self.device)
            # Define the norm. and activation for output of sub_kernels for edges
            self.edge_attr_relu = torch.nn.ReLU()
            self.edge_attr_bn1d    = torch.nn.BatchNorm1d(self.out_channels).to(self.device)

        # Learning Quantization vua MLPs
        quant_net_in_channels = self.pos_descr_dim if self.pos_descr_exists else self.in_channels # Default: Learn from Node Features
        quant_net_out_channels = self.num_sub_kernels # Outputs distribution over num subkernels (from which we select best kernel mapping)
        quant_net_left_channels = [ quant_net_in_channels ] + [ (i * quant_net_in_channels * self.quant_net_expansion) for i in range(1, self.quant_net_depth) ]
        quant_net_left_layers = []
        for ch_idx in range(len(quant_net_left_channels) - 1):
            in_channels  = quant_net_left_channels[ch_idx]
            out_channels = quant_net_left_channels[ch_idx + 1]
            quant_net_left_layers.extend([
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU()
            ])
        self.quant_net_left = torch.nn.Sequential(*quant_net_left_layers).to(self.device)

        # Define the layer that learns the quantization
        quant_net_right_channels = [ (i * quant_net_in_channels * self.quant_net_expansion) for i in range(self.quant_net_depth - 1, 0, -1) ] + [ quant_net_out_channels ]
        quant_net_right_layers = []
        for ch_idx in range(len(quant_net_right_channels) - 1):
            in_channels  = quant_net_right_channels[ch_idx]
            out_channels = quant_net_right_channels[ch_idx + 1]
            if ch_idx == len(quant_net_right_channels) - 2:
                # Last Layer is a mix of custom layers
                quant_net_right_layers.extend([
                    torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                    QuantizedActivation(num_sub_kernels=self.num_sub_kernels, activation="softmax")
                ])
                continue
            # Intermediate/Hidden Layers
            quant_net_right_layers.extend([
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                torch.nn.BatchNorm1d(out_channels),
                torch.nn.ReLU()
            ])
        self.quant_net_right = torch.nn.Sequential(*quant_net_right_layers).to(self.device)

        # Mixture Net
        if self.apply_mixture_net:
            # Define a conv1D layer that learns features across all nodes in all neighborhoods the same
            self.glob_feat_net = torch.nn.Sequential(
                                                        torch.nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, stride=1),
                                                        torch.nn.BatchNorm1d(self.out_channels),
                                                        torch.nn.ReLU()
                                                    ).to(self.device)

            # Define a conv1D layer to extract features in same hyper dimension as input for QuantResNet convolution layer
            self.glob_feat_extract_net = torch.nn.Sequential(
                                                            torch.nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, stride=1),
                                                            torch.nn.BatchNorm1d(self.out_channels),
                                                            torch.nn.ReLU()
                                                        ).to(self.device)

            # Define an MLP that mixes information from clustered features + global features
            # NOTE: An N-layer network to mix cluster-agnostic messages and cluster-specific messages 
            mixture_net_channels = [ self.out_channels ] + [ (i * self.out_channels * self.mixture_net_expansion) for i in range(1, self.mixture_net_depth) ] + [ self.out_channels ] 
            mixture_net_layers = []
            for ch_idx in range(len(mixture_net_channels) - 1):
                in_channels  = mixture_net_channels[ch_idx]
                out_channels = mixture_net_channels[ch_idx + 1]
                mixture_net_layers.extend([
                    torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                    torch.nn.BatchNorm1d(out_channels),
                    torch.nn.ReLU()
                ])
            self.mixture_net = torch.nn.Sequential(*mixture_net_layers).to(self.device)
        

    def forward(self, x, edge_index, pos:Optional[Tensor], edge_attr:Optional[Tensor]):
        # Information from neighborhood -> afforded us by the QuantResNet subkernels that learn only on a per neighborhood basis
        out_feats = self.propagate(edge_index=edge_index, x=x, pos=pos, edge_attr=edge_attr, aggr=self.aggr)  # [N, out_channels, label_dim] -- # Propagate the call through next stages of message passing framework

        # Conditionally enable mixing
        if self.apply_mixture_net:
            # Information from entire graph -> afforded us by the conv1D/Lin layer that learns on all the data (conv1D/LinLayer -> BN -> Relu)
            glob_out_feats = self.glob_feat_net(x.unsqueeze(dim=-1)).squeeze(dim=-1)
            # Residual with Global MLP conv. output
            glob_out_feats = glob_out_feats + self.glob_feat_extract_net(glob_out_feats.unsqueeze(dim=-1)).squeeze(dim=-1)
            # Apply mixture model to combine features 
            out_feats = self.mixture_net((out_feats + glob_out_feats).unsqueeze(dim=-1)).squeeze(dim=-1)

        # Return the learned features
        return out_feats
        

    def message(self, x_i, x_j, pos_i: Optional[Tensor], pos_j: Optional[Tensor], edge_attr: Optional[Tensor]):
        """
        x_i [num_edges, node_features]
        x_j [num_edges, node_features]
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        edge_attr [num_edges, edge_attr_dim]
        """
        # Learn the quantized neighborhoods from node attributes
        quant_source_feats, quant_target_feats = (pos_j, pos_i) if self.pos_descr_exists else (x_j, x_i)
        sub_kernels_neigh_quant = self.quant_net_right( self.quant_net_left(quant_source_feats.unsqueeze(dim=-1)) - self.quant_net_left(quant_target_feats.unsqueeze(dim=-1)) )
        
        # Compute transformed node messages
        num_edges, _ = x_j.shape
        x_j_transformed = torch.zeros((num_edges, self.out_channels)).to(self.device)
        for sub_kernel_idx in range(self.num_sub_kernels):
            # Get the mask of the specific kernel
            sub_kernel_mask = (sub_kernels_neigh_quant == sub_kernel_idx)
            # apply masking and message preparation
            x_j_filtered_transformed = self.nodes_sub_kernels[sub_kernel_idx](x_j[sub_kernel_mask].unsqueeze(dim=-1))
            x_j_filtered_transformed = x_j_filtered_transformed + self.use_shared_bias * self.nodes_shared_bias.unsqueeze(dim=-1)
            # Apply the edge feats/attrs as per feat. scalars
            if self.edge_attr_exists:
                # Downstream BN requires more than 1 sample in batch unless in eval mode
                if edge_attr[sub_kernel_mask].shape[0] == 1:
                    bn1d_in_training = self.edge_attr_bn1d.training
                    if bn1d_in_training: self.edge_attr_bn1d.eval()
                # Compute scalars and apply them to node features
                edge_attr_feats = self.edges_sub_kernels[sub_kernel_idx](edge_attr[sub_kernel_mask].unsqueeze(dim=-1))
                edge_attr_feats = edge_attr_feats + self.use_shared_bias * self.edges_shared_bias.unsqueeze(dim=-1)
                x_j_filtered_scalars = self.edge_attr_relu(self.edge_attr_bn1d(edge_attr_feats))
                x_j_filtered_transformed = x_j_filtered_transformed * x_j_filtered_scalars
                # [Conditional] Downstream BN (train mode) requires more than 1 sample in batch unless in eval mode
                if edge_attr[sub_kernel_mask].shape[0] == 1: 
                    if bn1d_in_training: self.edge_attr_bn1d.train() 
            # update the state of nodes
            x_j_transformed[sub_kernel_mask] = x_j_filtered_transformed.squeeze(dim=-1)

        # return the transformed node features
        return x_j_transformed # size => (num_edges, out_channels) => (num_edges, node_features)
        

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        return self.relu(self.bn1d(aggr_out))


    def init_sub_kernels(self, in_channels: int, out_channels: int,  num_subkernels: int, use_bias: bool = True):
        """
        Assists with initializing new MLPs/weights in our adaptable kernel for the convolution
        """
        assert num_subkernels >= 1, "num subkernels in the convolution must be > 0"

        sub_kernels = []
        for _ in range(num_subkernels):
            # Define the subkernel, optionally disabling bias depending on use
            lin_layer = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=use_bias)
            # Add to list of subkernels
            sub_kernels.append(lin_layer.to(self.device))

        # Return the set of convolving sub-kernels
        return torch.nn.ModuleList(sub_kernels)
