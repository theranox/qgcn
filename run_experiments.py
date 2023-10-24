import os
import math
import statistics
import wget
import yaml
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

from cnn.cnn_architectures import CNN
from sgcn.src.architectures import SGCN
from qgcn.qgcn_architectures import QGCN
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
        print(f"Dataset {selected_dataset_config['dataset_name'].upper()} for split: train-{train_size}, test-{test_size} doesn't exist")
        print(f"Downloading dataset ...")
        # Else: download the file before running experiment
        full_data_url = selected_dataset_config["download_url"][f"train_{train_size}_test_{test_size}"]
        wget.download(full_data_url, full_filepath)
        print(f"\nDownload complete ...")


# define the different datasets supported in sweep with their respective model config params
dataset_mapping = {
    "mnist": {
        "dataset_group": 'standard',
        "out_dim": 10,
        "dim_coor": 2,
        "in_channels": 1,
        "dataset_name": "mnist",
        "self_loops_included": True,
        "layers_num": 3,
        "model_dim": 32,
        "out_channels_1": 64,
        "hidden_channels": 32,
        "out_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        "qgcn_num_subkernels": 3 * 3, # same size as CNN kernel size
        "is_dataset_homogenous": True, # Homogenous means spatial location mask of the nodes do not change
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/da1sfgpr3sjoko5gboxm8/mnist.pkl?rlkey=rmj4ctryxovddq8u2bbd3im34&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/k6fudazul40xyl5kax203/mnist.pkl?rlkey=ucwsrfwc1jcne0lir9eyqmk1q&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/c6dj0qekv4n6mmwhbs2yf/mnist.pkl?rlkey=63uk506z9q3l6pvd04ahchueg&dl=0'
        }
    },
    "fashionmnist": {
        "dataset_group": 'standard',
        "out_dim": 10,
        "dim_coor": 2,
        "in_channels": 1,
        "dataset_name": "fashionmnist",
        "self_loops_included": True,
        "layers_num": 6,
        "model_dim": 32,
        "out_channels_1": 64,
        "hidden_channels": 32,
        "out_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        "qgcn_num_subkernels": 3 * 3, # same size as CNN kernel size
        "is_dataset_homogenous": True, # Homogenous means spatial location mask of the nodes do not change
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/dsoawuvc89etmgm5xyx0b/fashionmnist.pkl?rlkey=65kugyvhc83v9p7t0rgzcilxh&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/zy79stswbx3yapfyv0spn/fashionmnist.pkl?rlkey=sy7amb6ip5lu5t8w0vydjf64j&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/grqjcbpwm6rajnjob213q/fashionmnist.pkl?rlkey=czzodd35a84ltr0g3cp3ecl2x&dl=0'
        }
    },
    "cifar10": {
        "dataset_group": 'standard',
        "out_dim": 10,
        "dim_coor": 2,
        "in_channels": 3,
        "dataset_name": "cifar10",
        "self_loops_included": True,
        "layers_num": 9,
        "model_dim": 32,
        "out_channels_1": 64,
        "hidden_channels": 32,
        "out_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        "qgcn_num_subkernels": 3 * 3, # same size as CNN kernel size
        "is_dataset_homogenous": True, # Homogenous means spatial location mask of the nodes do not change
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/j74qvvagesf49paoli6ir/cifar10.pkl?rlkey=ve65fk8eit6wzg0ut39nrmpf9&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/l768t8v86xakf1n80e9q7/cifar10.pkl?rlkey=ig4dunlnwhuhrs30u9egan6p5&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/z7kdju2nzfnjojfgt8wz8/cifar10.pkl?rlkey=0herqwcsq0buzowbqzf3z4dcb&dl=0'
        }
    },
    "navier_stokes_binary": {
        "dataset_group": 'custom',
        "out_dim": 10,
        "dim_coor": 2,
        "in_channels": 1,
        "dataset_name": "navier_stokes_binary",
        "self_loops_included": False,
        "layers_num": 3,
        "model_dim": 32,
        "out_channels_1": 64,
        "hidden_channels": 32,
        "out_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        "qgcn_num_subkernels": 11,
        "is_dataset_homogenous": True, # Homogenous means spatial location mask of the nodes do not change
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/makidt3vilviwlgpea5ji/navier_stokes_binary.pkl?rlkey=qe6x025zr525nfeh5okbgsyek&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/1vnzlgqijietc6zgw3zsh/navier_stokes_binary.pkl?rlkey=z0nvx949jjk5g2t1fyjw0aqes&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/nw05b3glgjo5yd7mfq7vo/navier_stokes_binary.pkl?rlkey=a9034j6i91x6pu6fadztjests&dl=0'
        }
    },
    "navier_stokes_denary_1": {
        "dataset_group": 'custom',
        "out_dim": 10,
        "dim_coor": 2,
        "in_channels": 1,
        "dataset_name": "navier_stokes_denary_1",
        "self_loops_included": False,
        "layers_num": 3,
        "model_dim": 32,
        "out_channels_1": 64,
        "hidden_channels": 32,
        "out_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        "qgcn_num_subkernels": 11,
        "is_dataset_homogenous": True, # Homogenous means spatial location mask of the nodes do not change
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/poyc8hl05hyczm216s27t/navier_stokes_denary_1.pkl?rlkey=orx0p40mblec7klh9ab4z8res&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/ssagn1wjlc2jukns7zrnm/navier_stokes_denary_1.pkl?rlkey=th53vacnldcuej9mfx6s012j1&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/6c4fx1b5bdfnzytozypec/navier_stokes_denary_1.pkl?rlkey=dmz0dt8mnb1pqlgmpmkc1leop&dl=0'
        }
    },
    "navier_stokes_denary_2": {
        "dataset_group": 'custom',
        "out_dim": 10,
        "dim_coor": 2,
        "in_channels": 1,
        "dataset_name": "navier_stokes_denary_2",
        "self_loops_included": False,
        "layers_num": 3,
        "model_dim": 32,
        "out_channels_1": 64,
        "hidden_channels": 32,
        "out_channels": 64,
        "cnn_kernel_size": 3,
        "cnn_stride": 1,
        "cnn_padding": 1,
        "qgcn_num_subkernels": 11,
        "is_dataset_homogenous": True, # Homogenous means spatial location mask of the nodes do not change
        "download_url": { # Used if dataset split for this dataset doesn't exist
            "train_100_test_20": 'https://dl.dropboxusercontent.com/scl/fi/wz7zmag2cshoic70k6wo6/navier_stokes_denary_2.pkl?rlkey=smr3xcva3gruogmo0q8b8ejfz&dl=0',
            "train_1000_test_200": 'https://dl.dropboxusercontent.com/scl/fi/h1it9hfu572y7npmo4cjx/navier_stokes_denary_2.pkl?rlkey=3i63lk85c28nr74rl0mx4hode&dl=0',
            "train_10000_test_1000": 'https://dl.dropboxusercontent.com/scl/fi/t4r0vvsiksnpo3khg2via/navier_stokes_denary_2.pkl?rlkey=9urivllqql4hcfiwftsztmlw2&dl=0'
        }
    },
}


"""
Define sweep parameters
"""
datasets = {
    'standard': [ {"train":100,"test":20,"batch_size":16}, {"train":1000,"test":200,"batch_size":16}, {"train":10000,"test":1000,"batch_size":16} ],
    'custom':   [ {"train":100,"test":20,"batch_size":16}, {"train":1000,"test":200,"batch_size":16}, {"train":10000,"test":1000,"batch_size":16} ]
}
lrs    = {
    'standard': [10,  5,   1,   0.5,  0.1,  0.05,   0.01,  0.005,  0.001, 0.0001, 0.00001, 0.00005, 0.000001],
    'custom':   [10,  5,   1,   0.5,  0.1,  0.05,   0.01,  0.005,  0.001, 0.0001, 0.00001, 0.00005, 0.000001],
}      
epochs = {
    'standard': [100, 100, 150, 150,  300,  300,    400,   400,    500,   600,    800,     800,     800],
    'custom':   [100, 100, 100, 100,  100,  100,    100,   100,    100,   100,    100,     100,     100]
}
runs   = {
    'standard': [3,   3,   3,   3,    3,    3,      3,     3,      3,     3,      3,       3,       3],
    'custom':   [3,   3,   3,   3,    3,    3,      3,     3,      3,     3,      3,       3,       3]
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
def run_qgcn_experiments(dataset_groups: list[str] = ['standard', 'custom'], notraineval: bool = False):
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
                check_dataset_split_exists_else_download(dataset_split, selected_dataset_config)

                # Prep experiment name
                experiment_name = f"BATCH-RESULTS-ALL-DATASETS-{selected_dataset.capitalize()}_Summary"
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
                dataset_name           = selected_dataset_config["dataset_name"]
                self_loops_included    = selected_dataset_config["self_loops_included"]
                layers_num             = selected_dataset_config["layers_num"]
                model_dim              = selected_dataset_config["model_dim"]
                out_channels_1         = selected_dataset_config["out_channels_1"]
                dim_coor               = selected_dataset_config["dim_coor"]
                out_dim                = selected_dataset_config["out_dim"]
                in_channels            = selected_dataset_config["in_channels"]
                hidden_channels        = selected_dataset_config["hidden_channels"]
                out_channels           = selected_dataset_config["out_channels"]
                cnn_kernel_size        = selected_dataset_config["cnn_kernel_size"]
                cnn_stride             = selected_dataset_config["cnn_stride"]
                cnn_padding            = selected_dataset_config["cnn_padding"]
                qgcn_num_subkernels    = selected_dataset_config["qgcn_num_subkernels"]
                is_dataset_homogenous  = selected_dataset_config["is_dataset_homogenous"]

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
                        sgcn_model = SGCN(dim_coor=dim_coor,
                                        out_dim=out_dim,
                                        input_features=in_channels,
                                        layers_num=layers_num,
                                        model_dim=model_dim,
                                        out_channels_1=out_channels_1)

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
                        qgcn_model = QGCN(dim_coor=dim_coor,
                                        out_dim=out_dim,
                                        in_channels=in_channels,
                                        hidden_channels=hidden_channels,
                                        out_channels=out_channels,
                                        layers_num=layers_num,
                                        num_kernels=qgcn_num_subkernels,
                                        self_loops_included=self_loops_included,
                                        is_dataset_homogenous=is_dataset_homogenous, # determines whether to apply caching for kernel masks
                                        apply_spatial_scalars=False, # SGCN-like behavior; refer to code and paper for more details
                                        initializer_model=cnn_model, # comment this out to have independent initializations
                                        device=device)

                        # setup experiments to run
                        num_train, num_test = dataset_split["train"], dataset_split["test"]
                        experiment_id = f'_{run}_train_{num_train}_test_{num_test}_lr_{lr}_num_epochs_{num_epochs}'
                        experiment = Experiment(sgcn_model = sgcn_model,
                                                qgcn_model = qgcn_model,
                                                cnn_model = cnn_model,
                                                optim_params = optim_params,
                                                base_path = "./", 
                                                num_train = num_train,
                                                num_test = num_test,
                                                dataset_name = dataset_name,
                                                train_batch_size = batch_size,
                                                test_batch_size = batch_size,
                                                train_shuffle_data = True,
                                                test_shuffle_data = False,
                                                id = experiment_id) # mark this experiment ...

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


"""
Args prep
"""
import argparse
parser = argparse.ArgumentParser(description='Reproduce experimental results for QGCN vs SGCN vs CNN on standard and custom datasets')
parser.add_argument('-d', '--dataset', required=True, help='Type of dataset, i.e., standard / custom / all, to run experiment on')
parser.add_argument('--notraineval', action="store_true", default=False, help='Sets boolean whether we want to evaluate the models for training accuracy')

args = parser.parse_args()
notraineval = args.notraineval
dataset = args.dataset.strip().lower()
assert any([ dataset == t for t in ['standard', 'custom', 'all'] ])
dataset_groups = ['standard', 'custom'] if dataset.lower() == 'all' else [dataset]

# Validate dataset type is correct
print(f"Running experiments on {dataset.capitalize()} Datasets")
run_qgcn_experiments(dataset_groups=dataset_groups, notraineval=notraineval)
