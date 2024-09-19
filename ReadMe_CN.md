# 深度学习模型：图卷积网络预测修饰反应可能性

## 模型介绍

本项目采用深度学习模型，通过图卷积网络预测修饰反应前后分子发生的可能性。模型的输入是反应前后的一对分子SMILES格式字符串，输出是修饰反应能否发生的概率打分。

## 环境配置

### 系统要求

- Python 3.7 或更高版本
- pip or anaconda

### 安装依赖

在运行此模型之前，需要确保您的Python环境已经安装了以下库：

- torch
- torchvision
- torch_geometric
- sklearn
- rdkit
- matplotlib

对应的版本号为：
torch=1.12.1
rdkit=2022.3.5
torch-geometric= 2.1.0.post1
scikit-learn=0.22

您可以使用以下命令进行安装：

```bash
pip install torch torch_geometric sklearn rdkit matplotlib
```
## 训练模型数据准备

请确保您的数据集中的分子以SMILES格式表示，并存储在CSV或其他兼容格式中。
#### 训练集格式：
训练集文件表头需要包括描述，反应前分子SMILES，反应后分子SMILES，类别（1能反应，0不能反应）四个必要信息，示例如下：

| description | smile_before | smile_after | class|
|----------|----------|----------|----------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 1|
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 0|

#### 测试集格式：
测试集文件表头需要包括描述，反应前分子SMILES，反应后分子SMILES，三个必要信息，示例如下：

| description | smile_before | smile_after | 
|----------|----------|----------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 


其他额外信息不影响代码运行，具体格式见data文件夹中train_data_0125.txt和test_data_20240125.txt

## 模型训练

使用以下命令启动训练过程，或者直接通过python IDE如PyCharm运行main_train_gcn_model.py即可：
``` bash
python main_train_gcn_model.py
```
在main_train_gcn_model.py文件中，需要修改的信息是在代码最后的path_train，需要按数据准备部分的格式改成自己的训练集，构建好训练集文本之后放到data文件夹中，然后修改对应文件名称即可
``` bash
if __name__ == '__main__':

    path_train = r'data\train_data_0125.txt'
    modScore = ModGCNScore()
    modScore.readTrain(path_train)
    modScore.validation()

```
若顺利运行代码，在输出端会显示当前运行信息（模型训练过程中的警告可以忽略）：
1.读取训练集文件，输出数据集大小
2.按指定训练轮数进行训练，输出每一轮训练过程中训练集和验证集准确率指标
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
模型训练过程中的模型参数会保存到save文件夹中，格式为gcn_net_params-XXXX.pth，其中XXXX表示当前模型在验证集的准确率，一般情况下越高代表模型效果越好

## 模型预测

使用以下命令启动预测过程：
``` bash
python main_test_gcn_model.py
```
在main_test_gcn_model.py文件中，需要修改的信息是在代码最后的path_test和modScore.param_dic，分别代表要预测的结果文件和使用训练好的模型参数
1.path_test需要按数据准备部分的格式改成自己的训练集，构建好训练集文本之后放到data文件夹中，然后修改对应文件名称即可
2.modScore.param_dic去save文件夹中选择一个合适的预训练模型参数文件（一般建议选择验证准确率最高的参数文件，将参数文件名修改即可

``` bash
    
if __name__ == '__main__':

    path_test = r'data\test_data_20240125.txt'
    modScore = ModGCNScore()
    modScore.param_dic = r'./save/gcn_net_params-0.90886802295253.pth'
    modScore.readTest(path_test)
    modScore.predict()
```
若顺利运行代码，在输出端会显示当前运行信息（警告可以忽略）：
1.读取测试集文件，输出数据集大小
2.对所有结果进行预测，输出每一轮训练过程中训练集和验证集准确率指标
```bash
read predict result...
read done, valid result number and prepare to predict: 403
load pretrained model and predict...
predict done.
write predict result to  predict\2024-05-28-21-06-13.txt
```


## 模型预测结果解析
模型预测的结果会保存到predict文件夹中，格式为predict\XX-XX-XX-XX.txt，其中XX-XX-XX-XX表示当前模型预测的日期，内容是所有预测结果基础信息、模型打分和模型根据打分判断的类别
#### 结果文件格式
| description | smile_before | smile_after | predict|class
|----------|----------|----------|---------|---------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 0.999085545539856	|True
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 0.999725878238678	|True

模型的输出是一个概率打分，代表修饰反应发生的可能性。打分越高，反应发生的可能性越大


## 自动分析流程

使用以下命令启动预测过程：
``` bash
python AutoAnalysis.py
```

在AutoAnalysis.py文件中，需要修改的信息是在代码最后的list_info，需要将需要查找的delta mass匹配改成自己的结果，构建好之后放到data文件夹中，然后修改对应文件名称即可
``` bash
if __name__ == '__main__':

    process_num = 1
    path_modif = r'data\串联修饰库规则20231028.xlsx'
    dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info = func_load_mod_rule(path_modif)
    modTree = ModTree(dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info)
    
    list_info = read_delta_mass(r'D:\项目\郑大_构建修饰树\data.txt')
    path_out = r'data\data_search.txt.txt'
    f_w = open(path_out, 'w')

```
若顺利运行代码，在输出端会显示当前运行信息，代表对输入的delta mass找到了多组候选的修饰组合，进一步需要按照测试集的文件格式，计算各个后续修饰组合的反应前后SMILES结构，并将文件目录输入到程序中，程序会自动运行得到各个修饰组合的反应可能性打分：
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

下面是运行程序需要的delta mass匹配文件的格式和得到修饰组合后的预测格式：
#### delta mass匹配文件格式：
delta mass匹配文件表头需要包括需要查找的delta mass质量，对应的氨基酸位点，对应的氨基酸位置（是否在N-term或者C-term），示例如下：

| DeltaMass | aa | position|
|----------|----------|----------|
| -3.994915	| Glu | |
| 244.9382 | Asp | N-term|

#### 得到修饰组合后的预测格式：
| description | smile_before | smile_after | 
|----------|----------|----------|
| tri-Methylation| O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCC[N+](C)(C)C)N[CH2] | 
| Phospho | O=C([CH2])[C@H](CCCCN)N[CH2] | O=C([CH2])[C@H](CCCCNP(O)(O)=O)N[CH2] | 


## 注意事项
请确保您的数据已经转换为SMILES格式字符串。如果您在数据转换或模型运行过程中遇到任何问题，欢迎随时联系。
