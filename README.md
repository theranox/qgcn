# Quantized Graph Convolutional Networks (QGCNs)

# Dependencies
- PyTorch >= 1.1
- PyTorch geometric >= 1.1.2
* wget pkg is required for downloading datasets

# Running the code
To run QGCN experiments on standard dataset, run (from root of repo):
- Setup conda environment via: `conda env create --name qgcn --file=environment.yml`
- Activate the environment via: `conda activate qgcn`
- Run experiments by running the below:
```python
python3 run_experiments.py --dataset=standard
```

To run QGCN experiments on custom Navier Stokes FEM datasets, run (from root of repo):

```python
python3 run_experiments.py --dataset=custom
```

To run QGCN experiments on all datasets (both standard and custom), run (from root of repo):

```python
python3 run_experiments.py --dataset=all
```

* All results generated can be found inside `./Experiments` directory

## Other options
- You can also run the same experiments using the ipython notebook here: `./qgcn_experiments.ipynb`.
- To turn off evaluation of training accuracies on the model, you can add the flag `--notraineval`
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
