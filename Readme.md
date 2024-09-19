# Deep Learning Model: Graph Convolutional Network to Predict Modification Reaction Likelihood

## Model Introduction

This project employs a deep learning model using a graph convolutional network (GCN) to predict the likelihood of molecular changes before and after a modification reaction. The input to the model is a pair of molecules in SMILES format (before and after the reaction), and the output is a probability score indicating whether the modification reaction can occur.

## Environment Setup

### System Requirements

- Python 3.7 or higher
- pip or anaconda

### Install Dependencies

Before running this model, make sure your Python environment has the following libraries installed:

- torch
- torchvision
- torch_geometric
- sklearn
- rdkit
- matplotlib

The corresponding version numbers are: 
torch=1.12.1
rdkit=2022.3.5
torch-geometric= 2.1.0.post1
scikit-learn=0.22

You can install them using the following command:

```bash
pip install torch torch_geometric sklearn rdkit matplotlib
```
## Data Preparation for Model Training

Ensure that the molecules in your dataset are represented in SMILES format and stored in a CSV or other compatible format.
#### Training Set Format：
The training set file must include the following columns: description, SMILES of the molecule before the reaction, SMILES of the molecule after the reaction, and class (1 for a reaction, 0 for no reaction). Example:

| description | smile_before | smile_after | class|
|----------|----------|----------|----------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 1|
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 0|

#### Test Set Format:
The test set file must include the following columns: description, SMILES before the reaction, and SMILES after the reaction. Example:

| description | smile_before | smile_after | 
|----------|----------|----------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 


Other additional information will not affect code execution. Refer to the files in the data folder (e.g., train_data_0125.txt and test_data_20240125.txt) for specific formats.

## Model Training

To start training, use the following command or run main_train_gcn_model.py directly from a Python IDE like PyCharm:：
``` bash
python main_train_gcn_model.py
```
n the main_train_gcn_model.py file, modify the path_train variable at the end of the script to point to your training data file (constructed as described in the Data Preparation section), place the training data in the data folder, and update the file name accordingly:
``` bash
if __name__ == '__main__':

    path_train = r'data\train_data_0125.txt'
    modScore = ModGCNScore()
    modScore.readTrain(path_train)
    modScore.validation()

```
If the code runs successfully, the output will display information such as the dataset size and training accuracy for each epoch. Warnings during training can be ignored:

1. Reading the training file and displaying the dataset size.
2. Training for the specified number of epochs and showing training/validation accuracy metrics.

```bash
read train result...
read done, valid result number and prepare to train: 1381
1104 277
[[303 248]
 [265 288]]
epoch: 0     Train:    TPR 0.5207956600361664 TNR 0.5499092558983666 bacc 0.5353524579672665
2.551440715789795
[[ 89  46]
 [ 38 104]]
 epoch: 0     Valid:    TPR 0.7323943661971831 TNR 0.6592592592592592 bacc 0.6958268127282212
 ...
 ...

```
The model parameters will be saved in the save folder with the format gcn_net_params-XXXX.pth, where XXXX represents the validation accuracy. Generally, higher accuracy means a better-performing model.

## Model Prediction

To start the prediction process, use the following command:
``` bash
python main_test_gcn_model.py
```
In the main_test_gcn_model.py file, modify the path_test and modScore.param_dic variables at the end of the script. 
1. path_test should point to the test data, and modScore.
2. param_dic should be updated to use the saved model parameters from the save folder:


``` bash
    
if __name__ == '__main__':

    path_test = r'data\test_data_20240125.txt'
    modScore = ModGCNScore()
    modScore.param_dic = r'./save/gcn_net_params-0.90886802295253.pth'
    modScore.readTest(path_test)
    modScore.predict()
```
If the code runs successfully, it will display the dataset size and output prediction results:

1.Read the test set file and output the dataset size.
Predict results for all data and output the accuracy metrics for both the training set and validation set during each training round.

```bash
read predict result...
read done, valid result number and prepare to predict: 403
load pretrained model and predict...
predict done.
write predict result to  predict\2024-05-28-21-06-13.txt
```


## Model Prediction Results Analysis
The predicted results will be saved in the predict folder with the format predict\XX-XX-XX-XX.txt, where XX-XX-XX-XX represents the prediction date. The file will contain basic information about the predictions, including the probability scores and predicted classes.
#### Result File Format:
| description | smile_before | smile_after | predict|class
|----------|----------|----------|---------|---------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 0.999085545539856	|True
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 0.999725878238678	|True

The model output is a probability score representing the likelihood of the modification reaction occurring. Higher scores indicate a higher probability of reaction.


## Automated Analysis Process

To start the analysis process, use the following command:
``` bash
python AutoAnalysis.py
```

In the AutoAnalysis.py file, modify the list_info variable at the end of the script. This variable should match your delta mass results. After constructing it, place the data file in the data folder and update the file name accordingly:
``` bash
if __name__ == '__main__':

    process_num = 1
    path_modif = r'data\mod_rule_20231028.xlsx'
    dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info = func_load_mod_rule(path_modif)
    modTree = ModTree(dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info)
    
    list_info = read_delta_mass(r'D:\data.txt')
    path_out = r'data\data_search.txt.txt'
    f_w = open(path_out, 'w')

```
If the code runs successfully, it will display the delta mass and matching modification combinations found. You can further calculate SMILES structures before and after the reaction and input them into the program for automatic scoring.
``` bash
[-3.994915, 'Glu', '']
[['Glu->Leu/Ile substitution', 'dihydroxy', 'Formation of five membered aromatic heterocycle', 0.0], ['Glu->Lys substitution', 'alpha-amino adipic acid', 'Pyro-glu from E', 0.0], ['Glu->Lys substitution', 'alpha-amino adipic acid', 'Dehydration', 0.0], ['Glu->Pro substitution', 'Oxidation or Hydroxylation', 'formaldehyde adduct', -1.9073486e-06], ['Glu->Pro substitution', 'proline oxidation to pyroglutamic acid', 'Methylation', -1.9073486e-06], ['Glu->Thr substitution', 'Ethanolation', 'Formation of five membered aromatic heterocycle', 0.0], ['Pyro-Glu from E + Methylation', 0.0]]
[[2.0, 'Pyro-Glu from E + Methylation'], [1.0, 'Glu->Leu/Ile substitution', 'dihydroxy', 'Formation of five membered aromatic heterocycle'], [1.0, 'Glu->Lys substitution', 'alpha-amino adipic acid', 'Pyro-glu from E'], [1.0, 'Glu->Lys substitution', 'alpha-amino adipic acid', 'Dehydration'], [1.0, 'Glu->Thr substitution', 'Ethanolation', 'Formation of five membered aromatic heterocycle'], [0.9999980926513672, 'Glu->Pro substitution', 'Oxidation or Hydroxylation', 'formaldehyde adduct'], [0.9999980926513672, 'Glu->Pro substitution', 'proline oxidation to pyroglutamic acid', 'Methylation']]
79.7059714794159
7
0 [2.0, 'Pyro-Glu from E + Methylation']
1 [1.0, 'Glu->Leu/Ile substitution', 'dihydroxy', 'Formation of five membered aromatic heterocycle']
2 [1.0, 'Glu->Lys substitution', 'alpha-amino adipic acid', 'Pyro-glu from E']
3 [1.0, 'Glu->Lys substitution', 'alpha-amino adipic acid', 'Dehydration']
4 [1.0, 'Glu->Thr substitution', 'Ethanolation', 'Formation of five membered aromatic heterocycle']
5 [0.9999980926513672, 'Glu->Pro substitution', 'Oxidation or Hydroxylation', 'formaldehyde adduct']
6 [0.9999980926513672, 'Glu->Pro substitution', 'proline oxidation to pyroglutamic acid', 'Methylation']
please input path of SMILES info before and after of reaction
data/test_data_20240708res.txt
read predict result...
read done, valid result number and prepare to predict: 445
load pretrained model and predict...
predict done.
write predict result to  predict\2024-09-19-12-02-03.txt
```

Below is the format for the delta mass matching file and the predicted results after finding modification combinations.

#### Delta Mass Matching File Format：
The delta mass matching file must include the following columns: delta mass, amino acid, and position (if at N-term or C-term). Example:

| DeltaMass | aa | position|
|----------|----------|----------|
| -3.994915	| Glu | |
| 244.9382 | Asp | N-term|

#### Predicted Result Format:：
| description | smile_before | smile_after | 
|----------|----------|----------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 


## Note
Please ensure your data has been converted into SMILES format strings. If you encounter any issues during data conversion or model execution, feel free to contact us at any time.
