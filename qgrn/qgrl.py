import torch
import numpy as np
from torch import nn
from typing import Optional
from torch import Tensor
from torch_geometric.nn import MessagePassing

class QuantizedActivation(torch.nn.Module):
    __supported_activations = ["softmax", "gumbel_softmax"]

    def __init__(self, activation: str = "gumbel_softmax", tau: int = 0.001, hard_sampling: bool = True, hard_sampling_tau_sf: float = 0.1) -> Tensor:
        super(QuantizedActivation, self).__init__()
        self.hard_sampling = hard_sampling
        self.activation = activation.strip().lower()
        self.tau = (tau * hard_sampling_tau_sf) if (self.hard_sampling and (self.activation == "softmax")) else tau

        # Error handling for class
        assert self.activation in QuantizedActivation.__supported_activations, f"Target Activation - {activation} not in supported activations {QuantizedActivation.__supported_activations}"

    # Pickling requires actual function names 
    def custom_activation(self, x: Tensor):
        if self.activation == "softmax":
            return torch.nn.functional.softmax(x.squeeze(dim=-1) / self.tau, dim=-1).unsqueeze(dim=-1)
        else: # NOTE: gumbel softmax is default; all else being equal
            return torch.nn.functional.gumbel_softmax(x.squeeze(dim=-1), tau=self.tau, hard=self.hard_sampling, dim=-1).unsqueeze(dim=-1)

    def forward(self, x: Tensor):
        """
        x: [num_nodes, node_features] 
           This is the output (logits) 
           which we feed through this custom 
           activation layer
        """
        # Return output from custom activation
        return self.custom_activation(x)
    
    # # NOTE: Debug code: intercepting backward pass to check whether gradients are flowing
    # # Use below to inspect whether gradients are flowing through argmax
    # def backward(self, *args, **kwargs):
    #     import pdb; pdb.set_trace()
    #     return self.backward(*args, **kwargs)


class QGRL(MessagePassing):
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
        super(QGRL, self).__init__(aggr=aggr)

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
        self.nodes_feats_group_conv1d = self.init_sub_kernels_via_group_conv1d(self.in_channels, self.out_channels, self.num_sub_kernels, use_bias=(not self.use_shared_bias))
        if self.use_shared_bias: self.nodes_shared_bias = torch.nn.Parameter(torch.rand(1, self.out_channels), requires_grad=True).to(self.device)

        # Define subkernels for convolution-like operation on edge_attr if they exist
        if self.edge_attr_exists:
            assert isinstance(edge_attr_dim, int), "Num. Edge Attributes/Features (edge_attr_dim) is expected to be an int"
            self.edge_feats_group_conv1d = self.init_sub_kernels_via_group_conv1d(self.edge_attr_dim, self.out_channels, self.num_sub_kernels, use_bias=(not self.use_shared_bias))
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
                    QuantizedActivation(activation="gumbel_softmax", tau=0.01, hard_sampling=True)
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

            # Define a conv1D layer to extract features in same hyper dimension as input for QGRL convolution layer
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
        # Information from neighborhood -> afforded us by the QGRL subkernels that learn only on a per neighborhood basis
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
        receptive_field_masks = self.quant_net_right( self.quant_net_left(quant_source_feats.unsqueeze(dim=-1)) - self.quant_net_left(quant_target_feats.unsqueeze(dim=-1)) )

        # Apply group conv for node features
        out_node_features_transformed = self.apply_group_conv1d(x=x_j,
                                                                group_conv1d=self.nodes_feats_group_conv1d, 
                                                                receptive_field_masks=receptive_field_masks, 
                                                                use_shared_bias=self.use_shared_bias, 
                                                                shared_bias=self.nodes_shared_bias)

        # Conditionally handle edge attributes
        if self.edge_attr_exists:
            # Extract edge feature vectors
            out_edge_features_transformed = self.apply_group_conv1d(x=edge_attr,
                                                                    group_conv1d=self.edge_feats_group_conv1d, 
                                                                    receptive_field_masks=receptive_field_masks, 
                                                                    use_shared_bias=self.use_shared_bias, 
                                                                    shared_bias=self.edges_shared_bias)
            # Compute scalars and apply them to node features
            out_edge_scalars = self.edge_attr_relu(self.edge_attr_bn1d(out_edge_features_transformed))
            out_node_features_transformed = out_node_features_transformed * out_edge_scalars

        # Return output [dim -> (num_edges, out_features) ]
        return out_node_features_transformed # size => (num_edges, out_features)
    

    def apply_group_conv1d(self, group_conv1d: torch.nn.Conv1d, x: Tensor, receptive_field_masks: Tensor, use_shared_bias: bool, shared_bias: torch.nn.Parameter) -> Tensor:
        """
        x [num_edges, in_features]
        """
        # Reshape into broadcast-friendly shape
        N, S, E  = receptive_field_masks.shape
        receptive_field_masks_reshaped = receptive_field_masks.reshape((N, E, S)) # [num_edges, num_subkernels, 1] -> [num_edges, 1, num_subkernels]

        # Create the group conv input for both nodes and edges group convolutions
        x_unsq = x.unsqueeze(dim=-1) # [num_edges, in_features, 1]
        x_group_conv_input = x_unsq * receptive_field_masks_reshaped # [num_edges, in_features, 1] * [num_edges, 1, num_subkernels] => [num_edges, in_features, num_subkernels]
        x_group_conv_input_collapsed = x_group_conv_input.reshape((x_group_conv_input.shape[0], -1, 1)) # [num_edges, in_features * num_subkernels, 1]

        # Feed-forward through group convs
        out_group_conv1d = group_conv1d(x_group_conv_input_collapsed) # [num_edges, out_features * num_subkernels, 1]
        
        # Apply subkernel filter masks again to extract out subkernel specific outputs
        N, G, _ = out_group_conv1d.shape
        out_group_conv1d_expnd = out_group_conv1d.reshape((N, G // self.num_sub_kernels, self.num_sub_kernels)) # [num_edges, out_features, num_subkernels]
        out_filtered_group_conv1d_expnd = out_group_conv1d_expnd * receptive_field_masks_reshaped # [num_edges, out_features, num_subkernels] * [num_edges, 1, num_subkernels] => [num_edges, out_features, num_subkernels]
        out_features_transformed = out_filtered_group_conv1d_expnd.sum(dim=-1) + use_shared_bias * shared_bias # [num_edges, out_features, num_subkernels] -> [num_edges, out_features]

        # Return output [dim -> (num_edges, out_features) ]
        return out_features_transformed # [num_edges, out_features]


    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        return self.relu(self.bn1d(aggr_out))


    def init_sub_kernels_via_group_conv1d(self, in_channels: int, out_channels: int,  num_subkernels: int, use_bias: bool = True):
        """
        Assists with initializing new MLPs/weights in our adaptable kernel for the convolution
        """
        assert num_subkernels >= 1, "num subkernels in the convolution must be > 0"

        # Define group conv.: Optimizes on the for-loop through individual linear/conv1d subkernel convolutions
        return torch.nn.Conv1d(in_channels=(in_channels * num_subkernels), out_channels=(out_channels * num_subkernels), groups=num_subkernels, kernel_size=1, stride=1, bias=use_bias).to(self.device)
