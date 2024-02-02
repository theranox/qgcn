# Quantized Graph Convolutional Networks (QGCNs) and Quantized Residual Graph Networks (QRGNs)

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
To run CNN vs QGCN experiments on standard image datasets, run (from root of repo):
```python
python3 run_experiments_qgcn_validation_cnn_qgcn_sgcn.py --dataset=standard  --notraineval
```

To run QGCN vs SGCN experiments on custom Navier Stokes FEM graph datasets, run (from root of repo):
```python
python3 run_experiments_qgcn_validation_cnn_qgcn_sgcn.py --dataset=custom  --notraineval
```

To run QGCN vs SGCN experiments on all datasets (both standard and custom), run (from root of repo):
```python
python3 run_experiments_qgcn_validation_cnn_qgcn_sgcn.py --dataset=all  --notraineval
```

The below trains QRGN and SGCN on graphs with positional descriptors. 
```python
python3 ./run_experiments_qrgn_validation_sgcn_qrgn_models.py --dataset=all --notraineval
```

The below trains QRGN and GCNConv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qrgn_validation_gcnconv_qrgn_models.py --dataset=custom --notraineval
```

The below trains ChebConv and GraphConv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qrgn_validation_chebconv_graphconv_models.py --dataset=custom --notraineval
```

The below trains SGConv and GENConv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qrgn_validation_sgconv_genconv_models.py --dataset=custom --notraineval
```

The below trains GeneralConv and GATv2Conv on benchmark graph datasets without positional descriptors.
```python
python3 ./run_experiments_qrgn_validation_generalconv_gatv2conv_models.py --dataset=custom --notraineval
```

NOTE:
- All the executable python files above have a `dataset_mapping` variable, which is a collection of all datasets to train the models on.
- And the corresponding dataset splits to be trained on are defined inside the `datasets` dictionary.
- All results generated can be found inside `./Experiments` directory
- Adding the flag `--profilemodels` to the command-lines will print out the model parameters, flops and wall clock times etc.
- Adding the flag `--notraineval` to the command-lines will turn off evaluating training accuracies on the models


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

## NOTES
1. Datasets are stored on git-lfs storage servers
2. Results published in paper aggregated results from the stats below:
    - test_acc_avg_of_smaxs
    - test_acc_std_of_smaxs
    - train_acc_avg_of_smaxs
    - train_acc_std_of_smaxs
    - train_loss_avg_of_smaxs
    - train_loss_std_of_smaxs
2. Assuming early stopping of models, it'd be reasonable to have reported the max stats instead but we chose not to because  it doesn't capture the dynamics during training as well as a smoothened (averaging) method would
3. Notice in your experimental runs that max stats also supports the same equivalence claim, and much more strongly
than the average stats but due to reason 2. we decided to use average stats
