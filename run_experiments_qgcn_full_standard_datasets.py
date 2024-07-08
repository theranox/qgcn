import os
import math
import statistics
import wget
import yaml
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from cnn.architectures import CNN
from qgcn.architectures import QGCN
from qgrn.architectures import QGRN
from experiment import Experiment

# Empty cache
torch.cuda.empty_cache()

# current file directory
current_file_dirpath = os.path.dirname(os.path.realpath('__file__'))

# device type for all models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using cuda: ", torch.cuda.is_available(), "device")


"""
Confirms whether dataset exists else downloads from dropbox
"""
def check_dataset_split_exists_else_download(dataset_split: dict, selected_dataset_config: dict):  
    dataset_par_dirpath = os.path.join(".", "Dataset")
    if not os.path.exists(dataset_par_dirpath):
        os.mkdir(dataset_par_dirpath)
    rawgraph_subfolder_dirpath = os.path.join(dataset_par_dirpath, "RawGraph")
    if not os.path.exists(rawgraph_subfolder_dirpath):
        os.mkdir(rawgraph_subfolder_dirpath)
    # extract the dataset split
    train_size = dataset_split["train"]
    test_size = dataset_split["test"]
    dataset_split_folder = os.path.join(rawgraph_subfolder_dirpath, f"train_{train_size}_test_{test_size}")
    if not os.path.exists(dataset_split_folder):
        os.mkdir(dataset_split_folder)
    # get the filename and constr full filepath for downloaded file
    full_filepath = os.path.join(dataset_split_folder, f"{selected_dataset_config['dataset_name']}.pkl")
    if not os.path.exists(full_filepath):
        try:
            # Otherwise: attempt to download prepared pkts
            print(f"Dataset {selected_dataset_config['dataset_name'].upper()} for split: train-{train_size}, test-{test_size} doesn't exist")
            print(f"Downloading dataset ...")
            # Else: download the file before running experiment
            full_data_url = selected_dataset_config["download_url"].get(f"train_{train_size}_test_{test_size}", None)
            if full_data_url is None: 
                # if image_to_graph conversion exists, then we default to that
                if selected_dataset_config.get("image_to_graph_supported", False):
                    return True
                else: 
                    return False
            # Attempt downloading dataset
            wget.download(full_data_url, full_filepath)
            print(f"\nDownload complete ...")
            # Post-process
            if selected_dataset_config.get("image_to_graph_supported", False): selected_dataset_config["image_to_graph_supported"] = False
            return True
        except:
            print("Couldn't download dataset! Handing control to parent function")
            return False
    # Post-process
    if selected_dataset_config.get("image_to_graph_supported", False): selected_dataset_config["image_to_graph_supported"] = False
    # Return state
    return True


# define the different datasets supported in sweep with their respective model config params
dataset_mapping = {
    #################################################################################################################################################
    ################################################################ STANDARD DATASETS ##############################################################
    #################################################################################################################################################
    "mnist": {
        "dataset_group": 'standard',
        "dataset_name": "mnist",
        "layers_num": 3,
        "out_dim": 10, # num_classes
        "num_sub_kernels": 9, # <= max_node_degree
        "in_channels": 1, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": 2,
        "edge_attr_dim": -1,
        # Model specific params
        # cnn
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        # sgcn
        "sgcn_hidden_sf": 2, 
        "sgcn_out_sf": 1, 
        "sgcn_hidden_size": 7,
        # determines for QGCN whether to enable caching
        'is_dataset_homogenous': True,
        # Dataset loc
        "image_to_graph_supported": False, # Specifies whether there's ability to support image tensor to torch geo data conversion during runtime
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/da1sfgpr3sjoko5gboxm8/mnist.pkl?rlkey=rmj4ctryxovddq8u2bbd3im34&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/k6fudazul40xyl5kax203/mnist.pkl?rlkey=ucwsrfwc1jcne0lir9eyqmk1q&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/c6dj0qekv4n6mmwhbs2yf/mnist.pkl?rlkey=63uk506z9q3l6pvd04ahchueg&dl=0',
            "train_60000_test_10000": ''
        }
    },

    "fashionmnist": {
        "dataset_group": 'standard',
        "dataset_name": "fashionmnist",
        "layers_num": 6,
        "out_dim": 10, # num_classes
        "num_sub_kernels": 9, # <= max_node_degree
        "in_channels": 1, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": 2,
        "edge_attr_dim": -1,
        # Model specific params
        # cnn
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        # sgcn
        "sgcn_hidden_sf": 2, 
        "sgcn_out_sf": 1, 
        "sgcn_hidden_size": 5,
        # determines for QGCN whether to enable caching
        'is_dataset_homogenous': True,
        # Dataset loc
        "image_to_graph_supported": False, # Specifies whether there's ability to support image tensor to torch geo data conversion during runtime
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/dsoawuvc89etmgm5xyx0b/fashionmnist.pkl?rlkey=65kugyvhc83v9p7t0rgzcilxh&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/zy79stswbx3yapfyv0spn/fashionmnist.pkl?rlkey=sy7amb6ip5lu5t8w0vydjf64j&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/grqjcbpwm6rajnjob213q/fashionmnist.pkl?rlkey=czzodd35a84ltr0g3cp3ecl2x&dl=0',
            "train_60000_test_10000": ''
        }
    },

    "cifar10": {
        "dataset_group": 'standard',
        "dataset_name": "cifar10",
        "layers_num": 9,
        "out_dim": 10, # num_classes
        "num_sub_kernels": 9, # <= max_node_degree
        "in_channels": 3, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": 2,
        "edge_attr_dim": -1,
        # Model specific params
        # cnn
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        # sgcn
        "sgcn_hidden_sf": 2, 
        "sgcn_out_sf": 1, 
        "sgcn_hidden_size": 4,
        # determines for QGCN whether to enable caching
        'is_dataset_homogenous': True,
        # Dataset loc
        "image_to_graph_supported": False, # Specifies whether there's ability to support image tensor to torch geo data conversion during runtime
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/j74qvvagesf49paoli6ir/cifar10.pkl?rlkey=ve65fk8eit6wzg0ut39nrmpf9&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/l768t8v86xakf1n80e9q7/cifar10.pkl?rlkey=ig4dunlnwhuhrs30u9egan6p5&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/z7kdju2nzfnjojfgt8wz8/cifar10.pkl?rlkey=0herqwcsq0buzowbqzf3z4dcb&dl=0',
            "train_50000_test_10000": ''
        }
    },

    # ##################################################################################################################################################################
    # ######################################################################### STANDARD DATASETS ######################################################################
    # ##################################################################################################################################################################

    # # AIDS: 
    # #     num_classes: 2
    # #     max_node_degree:  7
    # #     node_features: pde_on-6
    # #     pos_exists: True
    # #     total_dataset: 2000     
    # #     splits:
    # #         - train_100_test_20
    # #         - train_1000_test_200
    # #         - train_1600_test_400
    # "aids_yes_pos": {
    #     "dataset_group": 'custom',
    #     "dataset_name": "aids",
    #     "layers_num": 3,
    #     "out_dim": 2, # num_classes
    #     "num_sub_kernels": 7, # <= max_node_degree
    #     "in_channels": 6, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": 2,
    #     "edge_attr_dim": -1,
    #     # Model specific params
    #     "sgcn_hidden_sf": 2, 
    #     "sgcn_out_sf": 2, 
    #     "sgcn_hidden_size": 4,
    #     # Dataset loc
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/u0znszatfokt6zahpbzat/aids.pkl?rlkey=5w9hd6avasooapyti7vyueje1&dl=0',
    #         "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/9zqupve86kggy9t3sotam/aids.pkl?rlkey=c5ukkdj76qh9e8vtofcp2gzgl&dl=0',
    #         "train_1600_test_400": 'https://dl.dropboxusercontent.com/scl/fi/gah4c4vpsqnm6qlfcaluu/aids.pkl?rlkey=cg95tdcqyixfjjnz8z8szk5si&dl=0'
    #     }
    # },

    # # COIL-DEL: 
    # #     num_classes: 100
    # #     max_node_degree:  15
    # #     node_features: pde_on-3
    # #     pos_exists: True
    # #     total_dataset: 3900
    # #     splits:
    # #         - train_500_test_100
    # #         - train_1000_test_200
    # #         - train_3200_test_700
    # "coil_del_yes_pos": {
    #     "dataset_group": 'custom',
    #     "dataset_name": "coil_del",
    #     "layers_num": 3,
    #     "out_dim": 100, # num_classes
    #     "num_sub_kernels": 7, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": 2,
    #     "edge_attr_dim": -1,
    #     # Model specific params
    #     "sgcn_hidden_sf": 2, 
    #     "sgcn_out_sf": 1, 
    #     "sgcn_hidden_size": 6,
    #     # Dataset loc
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_500_test_100": 'https://dl.dropboxusercontent.com/scl/fi/supb14nvmvsc52cq790n1/coil_del.pkl?rlkey=s1xk5ujxu3zjmkyeyhcweqgbp&dl=0',
    #         "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/n74d6lb5c1684scs984ld/coil_del.pkl?rlkey=huop699z0d55r4ru36m818mfu&dl=0',
    #         "train_3200_test_700": 'https://dl.dropboxusercontent.com/scl/fi/083lwxx146hpke009rc2m/coil_del.pkl?rlkey=doc4ddqdrofypnnxwxrj4z7zd&dl=0'
    #     }
    # },

    # # Letter-high: 
    # #     num_classes: 15
    # #     max_node_degree:  6
    # #     node_features: pde_on-3
    # #     pos_exists: True
    # #     total_dataset: 2250
    # #     splits:
    # #         - train_150_test_30
    # #         - train_1050_test_150
    # #         - train_1650_test_450
    # #         - train_1725_test_525
    # "letter_high_yes_pos": {       
    #     "dataset_group": 'custom',
    #     "dataset_name": "letter_high",
    #     "layers_num": 3,
    #     "out_dim": 15, # num_classes
    #     "num_sub_kernels": 6, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": 2,
    #     "edge_attr_dim": -1,
    #     # Model specific params
    #     "sgcn_hidden_sf": 2, 
    #     "sgcn_out_sf": 1, 
    #     "sgcn_hidden_size": 5,
    #     # Dataset loc
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/qeijfm8znt3gsg2eqhbcx/letter_high.pkl?rlkey=ypuk44y4t13ynkigt0iozypbu&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/4wp1tp05ihf2stlwir73y/letter_high.pkl?rlkey=59gw8aocinab3gfb6k5x4tmxh&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/p2jcjtcg70tuvku00pxqf/letter_high.pkl?rlkey=q7120ws7nyscelp3dal5o4wy0&dl=0',
    #         "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/9xu3u7ps22235jl7sjbh9/letter_high.pkl?rlkey=0rqdosvrhl4dozno5nscdj3ox&dl=0',
    #     }
    # },
    
    # # Letter-low:
    # #     num_classes: 15
    # #     max_node_degree:  5
    # #     node_features: pde_on-3
    # #     pos_exists: True
    # #     total_dataset: 2250
    # #     splits:
    # #         - train_150_test_30
    # #         - train_1050_test_150
    # #         - train_1650_test_450
    # #         - train_1725_test_525
    # "letter_low_yes_pos": {
    #     "dataset_group": 'custom',
    #     "dataset_name": "letter_low",
    #     "layers_num": 3,
    #     "out_dim": 15, # num_classes
    #     "num_sub_kernels": 5, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": 2,
    #     "edge_attr_dim": -1,
    #     # Model specific params
    #     "sgcn_hidden_sf": 2, 
    #     "sgcn_out_sf": 1, 
    #     "sgcn_hidden_size": 5,
    #     # Dataset loc
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/ni7cm0ju0i1aos6o7b7f6/letter_low.pkl?rlkey=ajnumwl6h15bnmkub5z6dn0yn&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/6i87436465zfhzaxd16rj/letter_low.pkl?rlkey=k1vyeozy3tpy5t2ws2u9jc6pq&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/wx7zhu26tiffo5cbcr1vm/letter_low.pkl?rlkey=j1ekozzpup3rv2n1oip58a4sy&dl=0',
    #         "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/grgyv5v402v8au0cv9702/letter_low.pkl?rlkey=ae8h2roth48jztkiilz05mhct&dl=0',
    #     }
    # },

    # # Letter-med: 
    # #     num_classes: 15
    # #     max_node_degree:  5
    # #     node_features: pde_on-3
    # #     pos_exists: True
    # #     total_dataset: 2250
    # #     splits:
    # #         - train_150_test_30
    # #         - train_1050_test_150
    # #         - train_1650_test_450
    # #         - train_1725_test_525
    # "letter_med_yes_pos": {
    #     "dataset_group": 'custom',
    #     "dataset_name": "letter_med",
    #     "layers_num": 3,
    #     "out_dim": 15, # num_classes
    #     "num_sub_kernels": 5, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": 2,
    #     "edge_attr_dim": -1,
    #     # Model specific params
    #     "sgcn_hidden_sf": 2, 
    #     "sgcn_out_sf": 1, 
    #     "sgcn_hidden_size": 5,
    #     # Dataset loc
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/0ue0ocmj70suvicj751lx/letter_med.pkl?rlkey=aqxyqmevkyru3k5it5yaw43sg&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/8f7j1z6lrywinw8ds0see/letter_med.pkl?rlkey=epg9jeej714aaz9i65ilxbqx2&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/eibutccv57m3xo4elck1h/letter_med.pkl?rlkey=om3q4s5qxyupsokmu1zc94cnx&dl=0',
    #         "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/md5815hms3imipkau7r2i/letter_med.pkl?rlkey=7kxh3ppky7wbmil2i9jx8yapn&dl=0',
    #     }
    # }
}


"""
Define sweep parameters
"""
datasets = {
    'standard':   [
        # -> All dataset splits
        # {"train": 100, "test": 20, "batch_size": 16},
        # {"train": 148, "test": 40, "batch_size": 64}, 
        # {"train": 150, "test": 30, "batch_size": 64}, 
        # {"train": 320, "test": 80, "batch_size": 64}, 
        # {"train": 480, "test": 120, "batch_size": 64},
        # {"train": 500, "test": 100, "batch_size": 64},
        # {"train": 850, "test": 250, "batch_size": 64},
        # {"train": 1000, "test": 100, "batch_size": 64},
        # {"train": 1000, "test": 200, "batch_size": 64},
        # {"train": 1050, "test": 150, "batch_size": 64},
        # {"train": 1600, "test": 400, "batch_size": 64},
        # {"train": 1650, "test": 450, "batch_size": 64},
        # {"train": 1650, "test": 495, "batch_size": 64},
        # {"train": 1725, "test": 525, "batch_size": 64},
        # {"train": 2000, "test": 500, "batch_size": 64},
        # {"train": 3200, "test": 700, "batch_size": 64},
        # {"train": 3400, "test": 900, "batch_size": 64},
        # {"train": 10000, "test": 1000, "batch_size": 64}

        # -> Test splits to validate scripts
        # {"train": 100, "test": 20, "batch_size": 16},
        # {"train": 150, "test": 30, "batch_size": 16},
        # {"train": 500, "test": 100, "batch_size": 16}

        # -> Target dataset splits to collect
        # {"train": 100, "test": 20, "batch_size": 32},
        # {"train": 1000, "test": 200, "batch_size": 32},
        # {"train": 1600, "test": 400, "batch_size": 32},
        # {"train": 1725, "test": 525, "batch_size": 32},
        # {"train": 3200, "test": 700, "batch_size": 32},
        # {"train": 10000, "test": 1000, "batch_size": 32},
        {"train": 50000, "test": 10000, "batch_size": 128},
        {"train": 60000, "test": 10000, "batch_size": 128}
    ],
    'custom':   [
        # -> All dataset splits
        # {"train": 100, "test": 20, "batch_size": 16},
        # {"train": 148, "test": 40, "batch_size": 64}, 
        # {"train": 150, "test": 30, "batch_size": 64}, 
        # {"train": 320, "test": 80, "batch_size": 64}, 
        # {"train": 480, "test": 120, "batch_size": 64},
        # {"train": 500, "test": 100, "batch_size": 64},
        # {"train": 850, "test": 250, "batch_size": 64},
        # {"train": 1000, "test": 100, "batch_size": 64},
        # {"train": 1000, "test": 200, "batch_size": 64},
        # {"train": 1050, "test": 150, "batch_size": 64},
        # {"train": 1600, "test": 400, "batch_size": 64},
        # {"train": 1650, "test": 450, "batch_size": 64},
        # {"train": 1650, "test": 495, "batch_size": 64},
        # {"train": 1725, "test": 525, "batch_size": 64},
        # {"train": 2000, "test": 500, "batch_size": 64},
        # {"train": 3200, "test": 700, "batch_size": 64},
        # {"train": 3400, "test": 900, "batch_size": 64},
        # {"train": 10000, "test": 1000, "batch_size": 64}

        # -> Test splits to validate scripts
        # {"train": 100, "test": 20, "batch_size": 16},
        # {"train": 150, "test": 30, "batch_size": 16},
        # {"train": 500, "test": 100, "batch_size": 16}

        # -> Target dataset splits to collect
        # {"train": 100, "test": 20, "batch_size": 32},
        # {"train": 1000, "test": 200, "batch_size": 32},
        # {"train": 1600, "test": 400, "batch_size": 32},
        # {"train": 1725, "test": 525, "batch_size": 32},
        # {"train": 3200, "test": 700, "batch_size": 32},
        # {"train": 10000, "test": 1000, "batch_size": 32},
        {"train": 50000, "test": 10000, "batch_size": 128},
        {"train": 60000, "test": 10000, "batch_size": 128}
    ]
}


lrs    = {
    'standard': [0.001, 0.01],
    'custom':   [0.001, 0.01],
}
epochs = {
    'standard': [200, 200],
    'custom':   [200, 200]
}
runs   = {
    'standard': [1,   1],
    'custom':   [1,   1]
}


"""
Helper function for collating results
Define function for handling collation
"""
def collate_stats(stats_name, max_stats, smoothened_stats):
  # collate the results to cache
  collated_stats_keys = [ f"{stats_name}_max_of_maxs",
                          f"{stats_name}_avg_of_maxs",
                          f"{stats_name}_std_of_maxs",
                          f"{stats_name}_max_of_smaxs",
                          f"{stats_name}_avg_of_smaxs",
                          f"{stats_name}_std_of_smaxs"  ]
  cnn_collated_results = { x: 0 for x in collated_stats_keys} 
  sgcn_collated_results = { x: 0 for x in collated_stats_keys} 
  qgcn_collated_results = { x: 0 for x in collated_stats_keys} 

  # save the results
  cnn_collated_results[f"{stats_name}_max_of_maxs"]  = round(max(max_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_max_of_smaxs"] = round(max(smoothened_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_avg_of_maxs"]  = round(statistics.mean(max_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_avg_of_smaxs"] = round(statistics.mean(smoothened_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_std_of_maxs"]  = round(0 if (len(max_stats["cnn"]) <= 1) else statistics.stdev(max_stats["cnn"]), 5)
  cnn_collated_results[f"{stats_name}_std_of_smaxs"] = round(0 if (len(smoothened_stats["cnn"]) <= 1) else statistics.stdev(smoothened_stats["cnn"]), 5)

  sgcn_collated_results[f"{stats_name}_max_of_maxs"]  = round(max(max_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_max_of_smaxs"] = round(max(smoothened_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_avg_of_maxs"]  = round(statistics.mean(max_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_avg_of_smaxs"] = round(statistics.mean(smoothened_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_std_of_maxs"]  = round(0 if (len(max_stats["sgcn"]) <= 1) else statistics.stdev(max_stats["sgcn"]), 5)
  sgcn_collated_results[f"{stats_name}_std_of_smaxs"] = round(0 if (len(smoothened_stats["sgcn"]) <= 1) else statistics.stdev(smoothened_stats["sgcn"]), 5)

  qgcn_collated_results[f"{stats_name}_max_of_maxs"]  = round(max(max_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_max_of_smaxs"] = round(max(smoothened_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_avg_of_maxs"]  = round(statistics.mean(max_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_avg_of_smaxs"] = round(statistics.mean(smoothened_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_std_of_maxs"]  = round(0 if (len(max_stats["qgcn"]) <= 1) else statistics.stdev(max_stats["qgcn"]), 5)
  qgcn_collated_results[f"{stats_name}_std_of_smaxs"] = round(0 if (len(smoothened_stats["qgcn"]) <= 1) else statistics.stdev(smoothened_stats["qgcn"]), 5)

  # Return results
  return cnn_collated_results, sgcn_collated_results, qgcn_collated_results


"""
Main function that runs experiments
Initiates experiments run on standard vs custom vs all datasets
"""
def run_experiments(dataset_groups: list[str] = ['standard', 'custom'], notraineval: bool = False, profile_models: bool = True):
    """
    SWEEPING Logic below
    Loops through the different sweep parameters to train different models
    """
    for dataset_group in dataset_groups:
        for dataset_split in datasets[dataset_group]: # loop over datasets
            # extract batch size which is peculiar to dataset split
            train_size = dataset_split.get('train', 0)
            test_size  = dataset_split.get('test', 0)
            batch_size = dataset_split.get('batch_size', 64)
            
            print(f"Dataset stats: train-{train_size}, test-{test_size}, batch_size-{batch_size}")
            # Inner loop goes over all datasets
            for selected_dataset, selected_dataset_config in dataset_mapping.items():
                # Skip all datasets that do not match the dataset group key
                if selected_dataset_config['dataset_group'] != dataset_group: continue

                # Check if dataset exists, if not then download
                if not check_dataset_split_exists_else_download(dataset_split, selected_dataset_config): continue

                # Prep experiment name
                experiment_name = f"BATCH-RESULTS-ALL-DATASETS-{selected_dataset.capitalize()}_QGCN_Reviewer_gFMP_Summary"
                experiments_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), "Experiments")
                experiment_result_filepath = os.path.join(experiments_dir, f'{"_".join(experiment_name.split(" "))}.yaml')
                averaging_window_width = 0.05 # fraction -> 5%
                results = {} # to hold results for saving
                if os.path.exists(experiment_result_filepath):
                    with open(experiment_result_filepath, "r") as file_stream:
                        results = yaml.safe_load(file_stream)
                        if results:
                            results = dict(results)
                        else:
                            results = {}

                dataset_name          = selected_dataset_config["dataset_name"]
                layers_num            = selected_dataset_config["layers_num"]
                out_dim               = selected_dataset_config["out_dim"]
                in_channels           = selected_dataset_config["in_channels"]
                hidden_channels       = selected_dataset_config["hidden_channels"]
                out_channels          = selected_dataset_config["out_channels"]
                num_sub_kernels       = selected_dataset_config["num_sub_kernels"]
                edge_attr_dim         = selected_dataset_config["edge_attr_dim"]
                pos_descr_dim         = selected_dataset_config["pos_descr_dim"]
                img2graph_support     = selected_dataset_config.get("image_to_graph_supported", False)
                self_loops_included   = selected_dataset_config.get("self_loops_included", True)
                is_dataset_homogenous = selected_dataset_config.get("is_dataset_homogenous", False)
                
                cnn_kernel_size        = selected_dataset_config["cnn_kernel_size"]
                cnn_stride             = selected_dataset_config["cnn_stride"]
                cnn_padding            = selected_dataset_config["cnn_padding"]

                print(f"Selected Dataset: {selected_dataset.upper()}")

                # Inner-Inner loop
                for i, lr in enumerate(lrs[dataset_group]): # loop over learning rates
                    optim_params = { "lr": lr }
                    num_epochs   = epochs[dataset_group][i]
                    num_runs     = runs[dataset_group][i]
                    # create the key for hashing into results
                    results_hash_key = f'train_{dataset_split["train"]}_test_{dataset_split["test"]}_lr_{lr}'
                    results[results_hash_key] = {}
                    # run stats
                    mean_train_loss = { "cnn": [], "sgcn": [], "qgcn": []}
                    smoothened_train_loss = { "cnn": [], "sgcn": [], "qgcn": []}
                    max_train_acc = { "cnn": [], "sgcn": [], "qgcn": []}
                    smoothened_train_acc = { "cnn": [], "sgcn": [], "qgcn": []}
                    max_test_acc = { "cnn": [], "sgcn": [], "qgcn": []}
                    smoothened_test_acc = { "cnn": [], "sgcn": [], "qgcn": []}
                    for run in range(num_runs): # loop over num runs
                        # cnn model
                        # NOTE: We only train equivalent CNN model only for Standard Dataset
                        cnn_model = None
                        if dataset_group == 'standard':
                            cnn_model = CNN(out_dim=out_dim,
                                            hidden_channels=hidden_channels,
                                            in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=cnn_kernel_size,
                                            stride=cnn_stride,
                                            layers_num=layers_num,
                                            padding=cnn_padding)

                        # qgcn model
                        qgcn_model = QGCN(dim_coor=pos_descr_dim,
                                        out_dim=out_dim,
                                        in_channels=in_channels,
                                        hidden_channels=hidden_channels,
                                        out_channels=out_channels,
                                        layers_num=layers_num,
                                        num_kernels=num_sub_kernels,
                                        self_loops_included=self_loops_included,
                                        is_dataset_homogenous=is_dataset_homogenous, # determines whether to apply caching for kernel masks
                                        apply_spatial_scalars=False, # SGCN-like behavior; refer to code and paper for more details
                                        initializer_model=cnn_model, # comment this out to have independent initializations
                                        device=device)
                        
                        # qgrn model
                        qgrn_model = QGRN(out_dim=out_dim,
                                          in_channels=in_channels,
                                          hidden_channels=hidden_channels,
                                          out_channels=out_channels,
                                          layers_num=layers_num,
                                          num_sub_kernels=num_sub_kernels,
                                          edge_attr_dim=edge_attr_dim,
                                          pos_descr_dim=pos_descr_dim,
                                          device=device)

                        # setup experiments to run
                        num_train, num_test = dataset_split["train"], dataset_split["test"]
                        experiment_id = f'_{run}_train_{num_train}_test_{num_test}_lr_{lr}_num_epochs_{num_epochs}'
                        experiment = Experiment(sgcn_model = None, # qgrn_model,
                                                qgcn_model = qgcn_model,
                                                cnn_model = None, # cnn_model,
                                                optim_params = optim_params,
                                                base_path = "./", 
                                                num_train = num_train,
                                                num_test = num_test,
                                                dataset_name = dataset_name,
                                                train_batch_size = batch_size,
                                                test_batch_size = batch_size,
                                                train_shuffle_data = True,
                                                test_shuffle_data = False,
                                                image_to_graph_supported = img2graph_support,
                                                profile_run=profile_models,
                                                id = experiment_id) # mark this experiment ...

                        if profile_models: break

                        # run the experiment ...
                        experiment.run(num_epochs=num_epochs, eval_training_set=(not notraineval)) # specify num epochs ...

                        # load collected stats during runs ...
                        (train_cnn_loss_array, train_qgcn_loss_array, train_sgcn_loss_array, \
                        train_cnn_acc_array, train_qgcn_acc_array, train_sgcn_acc_array, \
                        test_cnn_acc_array, test_qgcn_acc_array, test_sgcn_acc_array) = experiment.load_cached_results() # only accuracies on train and test sets ...
                        
                        # get the mean stats
                        mean_train_loss["cnn"].append(statistics.mean(train_cnn_loss_array))
                        mean_train_loss["sgcn"].append(statistics.mean(train_sgcn_loss_array))
                        mean_train_loss["qgcn"].append(statistics.mean(train_qgcn_loss_array))
                        
                        max_train_acc["cnn"].append(max(train_cnn_acc_array))
                        max_train_acc["sgcn"].append(max(train_sgcn_acc_array))
                        max_train_acc["qgcn"].append(max(train_qgcn_acc_array))
                        
                        max_test_acc["cnn"].append(max(test_cnn_acc_array))
                        max_test_acc["sgcn"].append(max(test_sgcn_acc_array))
                        max_test_acc["qgcn"].append(max(test_qgcn_acc_array))

                        # get the smoothened max test acc
                        num_averaging_window = int(math.ceil(averaging_window_width * num_epochs))
                        smoothened_train_loss["cnn"].append(statistics.mean(train_cnn_loss_array[-num_averaging_window:]))
                        smoothened_train_loss["sgcn"].append(statistics.mean(train_sgcn_loss_array[-num_averaging_window:]))
                        smoothened_train_loss["qgcn"].append(statistics.mean(train_qgcn_loss_array[-num_averaging_window:]))
                        
                        smoothened_train_acc["cnn"].append(statistics.mean(train_cnn_acc_array[-num_averaging_window:]))
                        smoothened_train_acc["sgcn"].append(statistics.mean(train_sgcn_acc_array[-num_averaging_window:]))
                        smoothened_train_acc["qgcn"].append(statistics.mean(train_qgcn_acc_array[-num_averaging_window:]))
                        
                        smoothened_test_acc["cnn"].append(statistics.mean(test_cnn_acc_array[-num_averaging_window:]))
                        smoothened_test_acc["sgcn"].append(statistics.mean(test_sgcn_acc_array[-num_averaging_window:]))
                        smoothened_test_acc["qgcn"].append(statistics.mean(test_qgcn_acc_array[-num_averaging_window:]))
                    
                    if profile_models: break

                    # get collated stats
                    train_loss_cnn_results, train_loss_sgcn_results, train_loss_qgcn_results = collate_stats("train_loss", mean_train_loss, smoothened_train_loss)
                    train_acc_cnn_results,  train_acc_sgcn_results,  train_acc_qgcn_results  = collate_stats("train_acc", max_train_acc, smoothened_train_acc)
                    test_acc_cnn_results,   test_acc_sgcn_results,   test_acc_qgcn_results   = collate_stats("test_acc", max_test_acc, smoothened_test_acc)
                    all_cnn_stats  = {**train_loss_cnn_results,  **train_acc_cnn_results,  **test_acc_cnn_results}
                    all_sgcn_stats = {**train_loss_sgcn_results, **train_acc_sgcn_results, **test_acc_sgcn_results}
                    all_qgcn_stats = {**train_loss_qgcn_results, **train_acc_qgcn_results, **test_acc_qgcn_results}

                    # save results into results obj
                    results[results_hash_key]["cnn"] = all_cnn_stats
                    results[results_hash_key]["sgcn"] = all_sgcn_stats
                    results[results_hash_key]["qgcn"] = all_qgcn_stats

                    # pickle the results
                    with open(experiment_result_filepath, "w") as file_stream:
                        yaml.dump(results, file_stream)

        if profile_models: break


"""
Args prep
"""
import argparse
parser = argparse.ArgumentParser(description='Reproduce experimental results for QGCN vs SGCN vs CNN on standard and custom datasets')
parser.add_argument('-d', '--dataset', required=True, help='Type of dataset, i.e., standard / custom / all, to run experiment on')
parser.add_argument('--notraineval', action="store_true", default=False, help='Sets boolean whether we want to evaluate the models for training accuracy')
parser.add_argument('--profilemodels', action="store_true", default=False, help='Sets boolean whether we want to profile the models for training')

args = parser.parse_args()
notraineval = args.notraineval
profile_models = args.profilemodels
dataset = args.dataset.strip().lower()
assert any([ dataset == t for t in ['standard', 'custom', 'all'] ])
dataset_groups = ['standard', 'custom'] if dataset.lower() == 'all' else [dataset]

# Validate dataset type is correct
print(f"Running experiments on {dataset.capitalize()} Datasets")
run_experiments(dataset_groups=dataset_groups, notraineval=notraineval, profile_models=profile_models)
