# Convolutional Neural Networks (CNNs) vs. Quantized Graph Convolutional Networks (QGCNs) vs. Quantized Residual Graph Networks (QGRNs)

# Dependencies
- PyTorch >= 1.1
- PyTorch geometric >= 1.1.2
- wget
- torch-geometric
- torch-cluster 
- torch-sparse 
- torch-scatter
*** Install the dependencies via pip e.g.,: 
```python
pip install wget torch-geometric torch-cluster torch-sparse torch-scatter
```

# Running the code
The results presented in the paper can be reproduced by either running the below commands directly from your command-line:
To run CNN vs QGCN vs QGRN experiments on standard image datasets, run (from root of repo):
```python
python3 run_experiments_*.py --dataset=standard/custom/all  --notraineval
```

For example:
To run QGCN vs SGCN experiments on custom graph datasets, run (from root of repo):
```python
python3 run_experiments_qgcn_validation_cnn_qgcn_sgcn.py --dataset=custom --notraineval
```

To run QGCN vs SGCN experiments on all datasets (both standard and custom), run (from root of repo):
```python
python3 run_experiments_qgcn_validation_cnn_qgcn_sgcn.py --dataset=all --notraineval
```

The below trains QGRN and SGCN on graphs with positional descriptors. 
```python
python3 ./run_experiments_qgrn_validation_sgcn_qgrn_models.py --dataset=all --notraineval
```

The below trains QGRN and GCNConv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qgrn_validation_gcnconv_qgrn_models.py --dataset=custom --notraineval
```

The below trains ChebConv and GraphConv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qgrn_validation_chebconv_graphconv_models.py --dataset=custom --notraineval
```

The below trains SGConv and GENConv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qgrn_validation_sgconv_genconv_models.py --dataset=custom --notraineval
```

The below trains GeneralConv and GATv2Conv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qgrn_validation_generalconv_gatv2conv_models.py --dataset=custom --notraineval
```


For full dataset (only applicable to standard image datasets: MNIST, FashionMNIST, CIFAR10), please run the below to first download and 
compile a pytorch geometric dataset before running the corresponding python experiment files (training the models with the said data):
```python
python3 ./generate_rawgraph_data.py --dataset=MNIST
```
```python
python3 ./generate_rawgraph_data.py --dataset=FashionMNIST
```
```python
python3 ./generate_rawgraph_data.py --dataset=CIFAR10
```

NOTE:
- All the executable python files above have a `dataset_mapping` variable, which is a collection of all datasets to train the models on.
- And the corresponding dataset splits to be trained on are defined inside the `datasets` dictionary.
- All results generated will be populated into `./Experiments` directory
- Adding the flag `--profilemodels` to the command-lines will print out the model parameters, flops and wall clock times etc.
- Adding the flag `--notraineval` to the command-lines will turn off evaluating training accuracies on the models
- All the experimental runs have been moved into the experimental_runs directory (other than that all prior guidelines & outlines apply)
- `--dataset=standard` will match all standard image datasets (i.e., MNIST, FashionMNIST, CIFAR10)
- `--dataset=custom` will match all custom datasets (i.e., datasets we post-processed from TUDatasets Benchmark)
- `--dataset=all` will match all datasets (i.e., both custom and standard)


## Other options
- You can also run the same experiments using the ipython notebooks in a google colab environment with GPUs
- Running the QGCN experiments from ipython will require specifying the relative path to the QGCN repository you cloned
- - This is detailed in the ipython notebooks
- For each model trained per learning rate (across different runs of the same learning rate), results are aggregated into:
    - ```lr```   - learning rate
    - ```max```  - absolute max accuracy/loss over all epochs of model train and test across different runs per lr  
    - ```smax``` - smoothened max where we average out the model's accuracy/loss over the last 5% of epochs per run
        - test_acc_avg_of_maxs: mean of the maxs across runs per learning rate of test data accuracy
        - test_acc_avg_of_smaxs: mean of the smaxs across runs per learning rate of test data accuracy
        - test_acc_max_of_maxs: max of the maxs across runs per learning rate of test data accuracy
        - test_acc_max_of_smaxs: max of the smaxs across runs per learning rate of test data accuracy
        - test_acc_std_of_maxs: standard deviation of the maxs across runs per learning rate of test data accuracy
        - test_acc_std_of_smaxs: standard deviation of the smaxs across runs per learning rate of test data accuracy
        - train_acc_avg_of_maxs: mean of the maxs across runs per learning rate of train data accuracy
        - train_acc_avg_of_smaxs: mean of the smaxs across runs per learning rate of train data accuracy
        - train_acc_max_of_maxs: max of the maxs across runs per learning rate of train data accuracy
        - train_acc_max_of_smaxs: max of the smaxs across runs per learning rate of train data accuracy
        - train_acc_std_of_maxs: standard deviation of the maxs across runs per learning rate of train data accuracy
        - train_acc_std_of_smaxs: standard deviation of the smaxs across runs per learning rate of train data accuracy
        - train_loss_avg_of_maxs: mean of the maxs across runs per learning rate of train data loss
        - train_loss_avg_of_smaxs: mean of the smaxs across runs per learning rate of train data loss
        - train_loss_max_of_maxs: max of the maxs across runs per learning rate of train data loss
        - train_loss_max_of_smaxs: max of the smaxs across runs per learning rate of train data loss
        - train_loss_std_of_maxs: standard deviation of the maxs across runs per learning rate of train data loss
        - train_loss_std_of_smaxs: standard deviation of the smaxs across runs per learning rate of train data loss
- As is shown in `./data_processing.py` from the `read_cached_graph_dataset()` the struct per dataset looks like below:
    -   {
        - "raw": {
            - "x_train_data": <data>,
            - "y_train_data": <data>,
            - "x_test_data" : <data>,
            - "y_test_data" : <data>
        - },
        - "geometric": {
            - "qgcn_train_data": <data>,
            - "qgcn_test_data" : <data>,
            - "sgcn_train_data": <data>,
            - "sgcn_test_data" : <data>
        - }
    -   }
    - NOTE: raw data is image data for training CNN models
        - also qgcn / sgcn are just placeholder names for graph datasets hence 
        - you'll find in the codebase that we read say sgcn/qgcn data to train qgrn model etc.
        - TLDR: graph datasets are not model-specific throughout this codebase & qgcn/sgcn datasets are exactly the same graph datasets

## NOTES
1. Datasets, if not present, will be pulled in from dropbox / AWS wherever applicable
2. Results published in paper aggregated results from the stats below:
    - test_acc_avg_of_smaxs
    - test_acc_std_of_smaxs
    - train_acc_avg_of_smaxs
    - train_acc_std_of_smaxs
    - train_loss_avg_of_smaxs
    - train_loss_std_of_smaxs
2. Assuming early stopping / check-pointing of models, it'd be reasonable to have reported the max stats instead but we chose not to because  it doesn't capture the dynamics during training as well as a smoothened (averaging) method does
3. Notice in the experimental runs that max stats also supports the same equivalence claim, and much more strongly
than the average stats but due to reason 2. we decided to use average stats
