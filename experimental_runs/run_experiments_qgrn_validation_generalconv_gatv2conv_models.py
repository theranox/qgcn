import os
import math
import statistics
import wget
import yaml
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from gnn.architectures import GeneralConvNet, GATv2ConvNet
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
            print(f"Dataset {selected_dataset_config['dataset_name'].upper()} for split: train-{train_size}, test-{test_size} doesn't exist")
            print(f"Downloading dataset ...")
            # Else: download the file before running experiment
            full_data_url = selected_dataset_config["download_url"].get(f"train_{train_size}_test_{test_size}", None)
            if full_data_url is None: return False
            # Attempt downloading dataset
            wget.download(full_data_url, full_filepath)
            print(f"\nDownload complete ...")
            return True
        except:
            print("Couldn't download dataset! Handing control to parent function")
            return False
    return True


# define the different datasets supported in sweep with their respective model config params
dataset_mapping = {
    # AIDS: 
    #     num_classes: 2
    #     max_node_degree:  7
    #     node_features: pde_on-6
    #     pos_exists: True
    #     total_dataset: 2000     
    #     splits:
    #         - train_100_test_20
    #         - train_1000_test_200
    #         - train_1600_test_400
    "aids_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "aids",
        "layers_num": 3,
        "out_dim": 2, # num_classes
        "num_sub_kernels": 7, # <= max_node_degree
        "in_channels": 6, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 4,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/u0znszatfokt6zahpbzat/aids.pkl?rlkey=5w9hd6avasooapyti7vyueje1&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/9zqupve86kggy9t3sotam/aids.pkl?rlkey=c5ukkdj76qh9e8vtofcp2gzgl&dl=0',
            "train_1600_test_400": 'https://dl.dropboxusercontent.com/scl/fi/gah4c4vpsqnm6qlfcaluu/aids.pkl?rlkey=cg95tdcqyixfjjnz8z8szk5si&dl=0'
        }
    },

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
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/u0znszatfokt6zahpbzat/aids.pkl?rlkey=5w9hd6avasooapyti7vyueje1&dl=0',
    #         "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/9zqupve86kggy9t3sotam/aids.pkl?rlkey=c5ukkdj76qh9e8vtofcp2gzgl&dl=0',
    #         "train_1600_test_400": 'https://dl.dropboxusercontent.com/scl/fi/gah4c4vpsqnm6qlfcaluu/aids.pkl?rlkey=cg95tdcqyixfjjnz8z8szk5si&dl=0'
    #     }
    # },

    # COIL-DEL: 
    #     num_classes: 100
    #     max_node_degree:  15
    #     node_features: pde_on-3
    #     pos_exists: True
    #     total_dataset: 3900
    #     splits:
    #         - train_500_test_100
    #         - train_1000_test_200
    #         - train_3200_test_700
    "coil_del_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "coil_del",
        "layers_num": 3,
        "out_dim": 100, # num_classes
        "num_sub_kernels": 7, # <= max_node_degree
        "in_channels": 3, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 3,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_500_test_100": 'https://dl.dropboxusercontent.com/scl/fi/supb14nvmvsc52cq790n1/coil_del.pkl?rlkey=s1xk5ujxu3zjmkyeyhcweqgbp&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/n74d6lb5c1684scs984ld/coil_del.pkl?rlkey=huop699z0d55r4ru36m818mfu&dl=0',
            "train_3200_test_700": 'https://dl.dropboxusercontent.com/scl/fi/083lwxx146hpke009rc2m/coil_del.pkl?rlkey=doc4ddqdrofypnnxwxrj4z7zd&dl=0'
        }
    },
    
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
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_500_test_100": 'https://dl.dropboxusercontent.com/scl/fi/supb14nvmvsc52cq790n1/coil_del.pkl?rlkey=s1xk5ujxu3zjmkyeyhcweqgbp&dl=0',
    #         "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/n74d6lb5c1684scs984ld/coil_del.pkl?rlkey=huop699z0d55r4ru36m818mfu&dl=0',
    #         "train_3200_test_700": 'https://dl.dropboxusercontent.com/scl/fi/083lwxx146hpke009rc2m/coil_del.pkl?rlkey=doc4ddqdrofypnnxwxrj4z7zd&dl=0'
    #     }
    # },

    # ENZYMES:
    #     num_classes: 6
    #     max_node_degree:  10
    #     node_features: pde_on-20
    #     pos_exists: False
    #     total_dataset: 600
    #     splits:
    #         - train_480_test_120
    "enzymes_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "enzymes",
        "layers_num": 3,
        "out_dim": 6, # num_classes
        "num_sub_kernels": 7, # <= max_node_degree
        "in_channels": 20, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 3,
        "general_out_sf": 3,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_480_test_120": 'https://dl.dropboxusercontent.com/scl/fi/71s43rvycjfxk4ubp3m7v/enzymes.pkl?rlkey=8bbt93zltcxlg4b00zjfo8152&dl=0',
        }
    },

    # # Fingerprint:
    # #   num_classes: 15
    # #   max_node_degree: 4
    # #   node_features: pde_on-3
    # #   pos_exists: True
    # #   edge_feats_dim: 2
    # #   total_dataset: 2149
    # #   splits:
    # #       - train_150_test_30
    # #       - train_1050_test_150
    # #       - train_1650_test_450
    # #       - train_1650_test_495
    # "fingerprint_no_pos_no_edge_feats": {        
    #     "dataset_group": 'custom',
    #     "dataset_name": "fingerprint",
    #     "layers_num": 3,
    #     "out_dim": 15, # num_classes
    #     "num_sub_kernels": 4, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": -1,
    #     "edge_attr_dim": -1,
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/161l8x67oglr6ostjik9s/fingerprint.pkl?rlkey=m2d474el113vn04xz2aqthbdt&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/r18isbxhvt9rw42v1e3gv/fingerprint.pkl?rlkey=ffoc8tzx3pbrfrdknoinnd5if&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/rstymnnql8ttsns5vbp1i/fingerprint.pkl?rlkey=obd317lucrcmisfdupzws21a2&dl=0',
    #         "train_1650_test_495": 'https://dl.dropboxusercontent.com/scl/fi/qwkdtwmb0p32bobm9z6u3/fingerprint.pkl?rlkey=0tegfnc5nl0zyr2zp4y2bxto4&dl=0',
    #     }
    # },

    # "fingerprint_no_pos_yes_edge_feats": {        
    #     "dataset_group": 'custom',
    #     "dataset_name": "fingerprint",
    #     "layers_num": 3,
    #     "out_dim": 15, # num_classes
    #     "num_sub_kernels": 4, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": -1,
    #     "edge_attr_dim": 2,
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/161l8x67oglr6ostjik9s/fingerprint.pkl?rlkey=m2d474el113vn04xz2aqthbdt&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/r18isbxhvt9rw42v1e3gv/fingerprint.pkl?rlkey=ffoc8tzx3pbrfrdknoinnd5if&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/rstymnnql8ttsns5vbp1i/fingerprint.pkl?rlkey=obd317lucrcmisfdupzws21a2&dl=0',
    #         "train_1650_test_495": 'https://dl.dropboxusercontent.com/scl/fi/qwkdtwmb0p32bobm9z6u3/fingerprint.pkl?rlkey=0tegfnc5nl0zyr2zp4y2bxto4&dl=0',
    #     }
    # },

    # "fingerprint_yes_pos_no_edge_feats": {        
    #     "dataset_group": 'custom',
    #     "dataset_name": "fingerprint",
    #     "layers_num": 3,
    #     "out_dim": 15, # num_classes
    #     "num_sub_kernels": 4, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": 2,
    #     "edge_attr_dim": -1,
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/161l8x67oglr6ostjik9s/fingerprint.pkl?rlkey=m2d474el113vn04xz2aqthbdt&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/r18isbxhvt9rw42v1e3gv/fingerprint.pkl?rlkey=ffoc8tzx3pbrfrdknoinnd5if&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/rstymnnql8ttsns5vbp1i/fingerprint.pkl?rlkey=obd317lucrcmisfdupzws21a2&dl=0',
    #         "train_1650_test_495": 'https://dl.dropboxusercontent.com/scl/fi/qwkdtwmb0p32bobm9z6u3/fingerprint.pkl?rlkey=0tegfnc5nl0zyr2zp4y2bxto4&dl=0',
    #     }
    # },

    # "fingerprint_yes_pos_yes_edge_feats": {        
    #     "dataset_group": 'custom',
    #     "dataset_name": "fingerprint",
    #     "layers_num": 3,
    #     "out_dim": 15, # num_classes
    #     "num_sub_kernels": 3, # <= max_node_degree
    #     "in_channels": 3, # node_features
    #     "hidden_channels": 32,
    #     "out_channels": 64,
    #     "pos_descr_dim": 2,
    #     "edge_attr_dim": 2,
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/161l8x67oglr6ostjik9s/fingerprint.pkl?rlkey=m2d474el113vn04xz2aqthbdt&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/r18isbxhvt9rw42v1e3gv/fingerprint.pkl?rlkey=ffoc8tzx3pbrfrdknoinnd5if&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/rstymnnql8ttsns5vbp1i/fingerprint.pkl?rlkey=obd317lucrcmisfdupzws21a2&dl=0',
    #         "train_1650_test_495": 'https://dl.dropboxusercontent.com/scl/fi/qwkdtwmb0p32bobm9z6u3/fingerprint.pkl?rlkey=0tegfnc5nl0zyr2zp4y2bxto4&dl=0',
    #     }
    # },

    # Frankenstein:
    #   num_classes: 2
    #   max_node_degree: 4
    #   node_features: pde_on-781, pde_off-779
    #   pos_exists: False
    #   total_dataset: 4337
    #   splits:
    #       - train:100_test:20
    #       - train:1000_test:200
    #       - train:2000_test:500
    #       - train:3400_test:900
    "frankenstein_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "frankenstein",
        "layers_num": 3,
        "out_dim": 2, # num_classes
        "num_sub_kernels": 4, # <= max_node_degree
        "in_channels": 781, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 3,
        "general_out_sf": 6,
        "general_hidden_sf": 8,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 2,
        "gatv2_hidden_sf": 4,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
          "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/ut4x5irjfor4357vno920/frankenstein.pkl?rlkey=6abjmsfwe6ehpr1u43oatr9ws&dl=0',
          "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/izejoggdc8p6ank3qxwfl/frankenstein.pkl?rlkey=65ojr9y2wzxa90sbp3mici943&dl=0',
          "train_2000_test_500": 'https://dl.dropboxusercontent.com/scl/fi/xil1ljva2gd2e5suq6he5/frankenstein.pkl?rlkey=7wbiy5jxdsf3cosf0r657a1c5&dl=0',
          "train_3400_test_900": 'https://dl.dropboxusercontent.com/scl/fi/3ubqhqwh7d4cvqkomkkpd/frankenstein.pkl?rlkey=jt7mgxgtiij62hpixq12okwi2&dl=0'
        }
    },

    # Letter-high: 
    #     num_classes: 15
    #     max_node_degree:  6
    #     node_features: pde_on-3
    #     pos_exists: True
    #     total_dataset: 2250
    #     splits:
    #         - train_150_test_30
    #         - train_1050_test_150
    #         - train_1650_test_450
    #         - train_1725_test_525
    "letter_high_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "letter_high",
        "layers_num": 3,
        "out_dim": 15, # num_classes
        "num_sub_kernels": 6, # <= max_node_degree
        "in_channels": 3, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 4,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 2,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 2,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/qeijfm8znt3gsg2eqhbcx/letter_high.pkl?rlkey=ypuk44y4t13ynkigt0iozypbu&dl=0',
            "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/4wp1tp05ihf2stlwir73y/letter_high.pkl?rlkey=59gw8aocinab3gfb6k5x4tmxh&dl=0',
            "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/p2jcjtcg70tuvku00pxqf/letter_high.pkl?rlkey=q7120ws7nyscelp3dal5o4wy0&dl=0',
            "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/9xu3u7ps22235jl7sjbh9/letter_high.pkl?rlkey=0rqdosvrhl4dozno5nscdj3ox&dl=0',
        }
    },

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
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/qeijfm8znt3gsg2eqhbcx/letter_high.pkl?rlkey=ypuk44y4t13ynkigt0iozypbu&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/4wp1tp05ihf2stlwir73y/letter_high.pkl?rlkey=59gw8aocinab3gfb6k5x4tmxh&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/p2jcjtcg70tuvku00pxqf/letter_high.pkl?rlkey=q7120ws7nyscelp3dal5o4wy0&dl=0',
    #         "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/9xu3u7ps22235jl7sjbh9/letter_high.pkl?rlkey=0rqdosvrhl4dozno5nscdj3ox&dl=0',
    #     }
    # },

    # Letter-low:
    #     num_classes: 15
    #     max_node_degree:  5
    #     node_features: pde_on-3
    #     pos_exists: True
    #     total_dataset: 2250
    #     splits:
    #         - train_150_test_30
    #         - train_1050_test_150
    #         - train_1650_test_450
    #         - train_1725_test_525
    "letter_low_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "letter_low",
        "layers_num": 3,
        "out_dim": 15, # num_classes
        "num_sub_kernels": 5, # <= max_node_degree
        "in_channels": 3, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 4,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 2,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 2,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/ni7cm0ju0i1aos6o7b7f6/letter_low.pkl?rlkey=ajnumwl6h15bnmkub5z6dn0yn&dl=0',
            "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/6i87436465zfhzaxd16rj/letter_low.pkl?rlkey=k1vyeozy3tpy5t2ws2u9jc6pq&dl=0',
            "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/wx7zhu26tiffo5cbcr1vm/letter_low.pkl?rlkey=j1ekozzpup3rv2n1oip58a4sy&dl=0',
            "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/grgyv5v402v8au0cv9702/letter_low.pkl?rlkey=ae8h2roth48jztkiilz05mhct&dl=0',
        }
    },

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
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/ni7cm0ju0i1aos6o7b7f6/letter_low.pkl?rlkey=ajnumwl6h15bnmkub5z6dn0yn&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/6i87436465zfhzaxd16rj/letter_low.pkl?rlkey=k1vyeozy3tpy5t2ws2u9jc6pq&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/wx7zhu26tiffo5cbcr1vm/letter_low.pkl?rlkey=j1ekozzpup3rv2n1oip58a4sy&dl=0',
    #         "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/grgyv5v402v8au0cv9702/letter_low.pkl?rlkey=ae8h2roth48jztkiilz05mhct&dl=0',
    #     }
    # },

    # Letter-med: 
    #     num_classes: 15
    #     max_node_degree:  5
    #     node_features: pde_on-3
    #     pos_exists: True
    #     total_dataset: 2250
    #     splits:
    #         - train_150_test_30
    #         - train_1050_test_150
    #         - train_1650_test_450
    #         - train_1725_test_525
    "letter_med_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "letter_med",
        "layers_num": 3,
        "out_dim": 15, # num_classes
        "num_sub_kernels": 5, # <= max_node_degree
        "in_channels": 3, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 4,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 2,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 2,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/0ue0ocmj70suvicj751lx/letter_med.pkl?rlkey=aqxyqmevkyru3k5it5yaw43sg&dl=0',
            "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/8f7j1z6lrywinw8ds0see/letter_med.pkl?rlkey=epg9jeej714aaz9i65ilxbqx2&dl=0',
            "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/eibutccv57m3xo4elck1h/letter_med.pkl?rlkey=om3q4s5qxyupsokmu1zc94cnx&dl=0',
            "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/md5815hms3imipkau7r2i/letter_med.pkl?rlkey=7kxh3ppky7wbmil2i9jx8yapn&dl=0',
        }
    },

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
    #     "download_url": { # Used if dataset split for this dataset doesn't exist
    #         "train_150_test_30": 'https://dl.dropboxusercontent.com/scl/fi/0ue0ocmj70suvicj751lx/letter_med.pkl?rlkey=aqxyqmevkyru3k5it5yaw43sg&dl=0',
    #         "train_1050_test_150": 'https://dl.dropboxusercontent.com/scl/fi/8f7j1z6lrywinw8ds0see/letter_med.pkl?rlkey=epg9jeej714aaz9i65ilxbqx2&dl=0',
    #         "train_1650_test_450": 'https://dl.dropboxusercontent.com/scl/fi/eibutccv57m3xo4elck1h/letter_med.pkl?rlkey=om3q4s5qxyupsokmu1zc94cnx&dl=0',
    #         "train_1725_test_525": 'https://dl.dropboxusercontent.com/scl/fi/md5815hms3imipkau7r2i/letter_med.pkl?rlkey=7kxh3ppky7wbmil2i9jx8yapn&dl=0',
    #     }
    # },

    # Mutag:
    #   num_classes: 2
    #   max_node_degree: 5
    #   node_features: pde_on-2, pde_off-2
    #   pos_exists: False
    #   total_dataset: 188
    #   splits:
    #       - train:100_test:20
    #       - train:148_test:40
    "mutag_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "mutag",
        "layers_num": 3,
        "out_dim": 2, # num_classes
        "num_sub_kernels": 5, # <= max_node_degree
        "in_channels": 2, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 4,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/5i18wan10o2wc7fjkdvj2/mutag.pkl?rlkey=p91qo0nqgdfyso65zduuayjce&dl=0',
            "train_148_test_40": 'https://dl.dropboxusercontent.com/scl/fi/xz1qqnuios20q7m8t9b82/mutag.pkl?rlkey=z65l75s7sf2ltf3r91jj2wrdy&dl=0'
        }
    },

    # Mutagenicity:
    #     num_classes: 2
    #     max_node_degree:  5
    #     node_features: pde_on-2
    #     pos_exists: False
    #     total_dataset: 4337
    #     splits:
    #         - train_100_test_20
    #         - train_1000_test_200
    #         - train_2000_test_500
    #         - train_3400_test_900
    "mutagenicity_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "mutagenicity",
        "layers_num": 3,
        "out_dim": 2, # num_classes
        "num_sub_kernels": 7, # <= max_node_degree
        "in_channels": 2, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 4,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/dgf8eg0fu4fea79yts7aq/mutagenicity.pkl?rlkey=oxt6heyk57wg0okww9wmsziul&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/6congycx03pz0wfa3dnqn/mutagenicity.pkl?rlkey=m95cuxllivykc4zkuj501sp69&dl=0',
            "train_2000_test_500": 'https://dl.dropboxusercontent.com/scl/fi/h2qxegbpdz5q0k06cs551/mutagenicity.pkl?rlkey=ds5weycseottry0tkmsmkfl8r&dl=0',
            "train_3400_test_900": 'https://dl.dropboxusercontent.com/scl/fi/8mkjfm1wjk9sqfj076zir/mutagenicity.pkl?rlkey=saquia4aan51t14g0n0a1rje5&dl=0',
        }
    },

    # PROTEINS:
    #     num_classes: 2
    #     max_node_degree:  26
    #     node_features: pde_on-3
    #     total_dataset: 1113
    #     splits:
    #         - train_100_test_20
    #         - train_850_test_250
    #         - train_1000_test_100
    "proteins_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "proteins",
        "layers_num": 3,
        "out_dim": 2, # num_classes
        "num_sub_kernels": 7, # <= max_node_degree
        "in_channels": 3, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 4,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/rkxvnyzhmxb4ce3i9tg3a/proteins.pkl?rlkey=44sqijlcxg9eddfej5ve7njcd&dl=0',
            "train_850_test_250": 'https://dl.dropboxusercontent.com/scl/fi/zbof8j650p7tna6qu5iyi/proteins.pkl?rlkey=nu1pwc641rfdntbiy2ml8zdr6&dl=0',
            "train_1000_test_100": 'https://dl.dropboxusercontent.com/scl/fi/7vmhiaujh7vc1klmsphhx/proteins.pkl?rlkey=twvrxidtqaqg1vrbt0zkhxvge&dl=0'
        }
    },

    # PROTEINS-Full:
    #     num_classes: 2
    #     max_node_degree:  26
    #     node_features: pde_on-31
    #     total_dataset: 1113
    #     splits:
    #         - train_100_test_20
    #         - train_850_test_250
    #         - train_1000_test_100
    "proteins_full_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "proteins_full",
        "layers_num": 3,
        "out_dim": 2, # num_classes
        "num_sub_kernels": 7, # <= max_node_degree
        "in_channels": 31, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 2,
        "general_out_sf": 5,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/0s3pq39bo18exm2gfz8ay/proteins_full.pkl?rlkey=0hxg4roi8rh2n89dzh53axl13&dl=0',
            "train_850_test_250": 'https://dl.dropboxusercontent.com/scl/fi/1e8iqcga032mx6u44vrf1/proteins_full.pkl?rlkey=u6in9pdjpniqptwr39d7i8438&dl=0',
            "train_1000_test_100": 'https://dl.dropboxusercontent.com/scl/fi/3cyvyyan0ttbywldsq1c1/proteins_full.pkl?rlkey=5sw1hnik2or76vrzxth4cca9n&dl=0'
        }
    },

    # Synthie:
    #     num_classes: 4
    #     max_node_degree:  21
    #     node_features: pde_on-16, pde_off-16
    #     total_dataset: 400
    #     splits:
    #         - train:100_test:20
    #         - train:320_test:80
    "synthie_no_pos": {
        "dataset_group": 'custom',
        "dataset_name": "synthie",
        "layers_num": 3,
        "out_dim": 4, # num_classes
        "num_sub_kernels": 7, # <= max_node_degree
        "in_channels": 16, # node_features
        "hidden_channels": 32,
        "out_channels": 64,
        "pos_descr_dim": -1,
        "edge_attr_dim": -1,
        # Model specific params
        "general_hidden_heads": 3,
        "general_out_sf": 3,
        "general_hidden_sf": 2,
        "gatv2_hidden_heads": 3,
        "gatv2_out_sf": 1,
        "gatv2_hidden_sf": 1,
        # Dataset loc
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/eyw8n9pge2ov784dzm4xn/synthie.pkl?rlkey=6jd12ckld2oowjjo4njqapegb&dl=0',
            "train_320_test_80": 'https://dl.dropboxusercontent.com/scl/fi/sadg4ztpryrmqq0ehqa4n/synthie.pkl?rlkey=4mce8t33mxmvyh985t91bmnte&dl=0',
        }
    }
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
        # {"train": 480, "test": 120, "batch_size": 16},
        # {"train": 500, "test": 100, "batch_size": 16},

        # -> Target dataset splits to collect
        {"train": 148, "test": 40, "batch_size": 32}, 
        {"train": 320, "test": 80, "batch_size": 32},
        {"train": 480, "test": 120, "batch_size": 32},
        {"train": 1000, "test": 100, "batch_size": 32},
        {"train": 1600, "test": 400, "batch_size": 32},
        {"train": 1650, "test": 495, "batch_size": 32},
        {"train": 1725, "test": 525, "batch_size": 32},
        {"train": 3200, "test": 700, "batch_size": 32},
        {"train": 3400, "test": 900, "batch_size": 32},
    ]
}

lrs    = {
    'standard': [0.1, 0.01, 0.001, 0.0001],
    'custom':   [0.1, 0.01, 0.001, 0.0001],
}
epochs = {
    'standard': [500, 500, 500, 500],
    'custom':   [500, 500, 500, 500]
}
runs   = {
    'standard': [3,   3,   3,   3],
    'custom':   [3,   3,   3,   3]
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
                experiment_name = f"BATCH-RESULTS-ALL-DATASETS-{selected_dataset.capitalize()}_GeneralConv_GATv2Conv_Summary"
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
                            
                # Load the required params for this dataset
                dataset_name       = selected_dataset_config["dataset_name"]
                layers_num         = selected_dataset_config["layers_num"]
                out_dim            = selected_dataset_config["out_dim"]
                in_channels        = selected_dataset_config["in_channels"]
                hidden_channels    = selected_dataset_config["hidden_channels"]
                out_channels       = selected_dataset_config["out_channels"]
                num_sub_kernels    = selected_dataset_config["num_sub_kernels"]
                edge_attr_dim      = selected_dataset_config["edge_attr_dim"]
                pos_descr_dim      = selected_dataset_config["pos_descr_dim"]

                # Model specific params
                general_hidden_heads = selected_dataset_config.get("general_hidden_heads", 1)
                general_out_sf       = selected_dataset_config.get("general_out_sf", 1)
                general_hidden_sf    = selected_dataset_config.get("general_hidden_sf", 1)

                gatv2_hidden_heads = selected_dataset_config.get("gatv2_hidden_heads", 1)
                gatv2_out_sf       = selected_dataset_config.get("gatv2_out_sf", 1)
                gatv2_hidden_sf    = selected_dataset_config.get("gatv2_hidden_sf", 1)

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
                        # initialize the models
                        generalconv_model = GeneralConvNet(out_dim=out_dim, 
                                                        input_features=in_channels, 
                                                        output_channels=out_channels, 
                                                        layers_num=layers_num, 
                                                        model_dim=hidden_channels,
                                                        hidden_sf=general_hidden_sf, 
                                                        out_sf=general_out_sf, 
                                                        hidden_heads=general_hidden_heads)

                        gatv2conv_model = GATv2ConvNet(out_dim=out_dim, 
                                                    input_features=in_channels,
                                                    output_channels=out_channels,
                                                    layers_num=layers_num,
                                                    model_dim=hidden_channels,
                                                    hidden_sf=gatv2_hidden_sf, 
                                                    out_sf=gatv2_out_sf,
                                                    hidden_heads=gatv2_hidden_heads)

                        # setup experiments to run
                        num_train, num_test = dataset_split["train"], dataset_split["test"]
                        experiment_id = f'_{run}_train_{num_train}_test_{num_test}_lr_{lr}_num_epochs_{num_epochs}'
                        experiment = Experiment(sgcn_model = generalconv_model,
                                                qgcn_model = gatv2conv_model,
                                                cnn_model = None,
                                                optim_params = optim_params,
                                                base_path = "./", 
                                                num_train = num_train,
                                                num_test = num_test,
                                                dataset_name = dataset_name,
                                                train_batch_size = batch_size,
                                                test_batch_size = batch_size,
                                                train_shuffle_data = True,
                                                test_shuffle_data = False,
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
