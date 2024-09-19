import pandas as pd
import numpy as np
import multiprocessing
import time
Global_ppm_threshold = 5

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from model_Transformer import TransformerEncoder
from model_gcn_new import CCPGraph
from feature_graph import Mol2Graph
from rdkit.Chem import AllChem as Chem
import numpy as np
import random
import matplotlib.pyplot as plt

seed = 5
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.


class DataFeat(object):
    def __init__(self, **kwargs):
        for k in kwargs:
            self.__dict__[k] = kwargs[k]


class ModGCNScore:

    def __init__(self):

        self.net = CCPGraph()
        self.param_dic = r'./best_model/gcn_net_params-0.9550913115661317.pth'
        self.data_info = []
        self.feature = []

    def createGraph(self, i_smiles1, i_smiles2, i_label):

        mol1 = Chem.MolFromSmiles(i_smiles1)
        mol2 = Chem.MolFromSmiles(i_smiles2)

        g1 = Mol2Graph(mol1)
        g2 = Mol2Graph(mol2)
        x = np.concatenate([g1.x, g2.x], axis=0)
        edge_feats = np.concatenate([g1.edge_feats, g2.edge_feats], axis=0)
        e_idx2 = g2.edge_idx + g1.node_num
        edge_index = np.concatenate([g1.edge_idx, e_idx2], axis=0)
        tmp_graph = DataFeat(x=x, edge_feats=edge_feats, edge_index=edge_index, y=np.array([i_label], dtype=np.float))

        return tmp_graph

    def readTrain(self, path_train: str):

        with open(path_train, 'rb') as f:
            lines = f.read().decode(encoding='utf-8').split('\r\n')
        table_list = lines[0].split('\t')

        position_smiles1 = table_list.index('smile_before')
        position_smiles2 = table_list.index('smile_after')
        position_class = table_list.index('class')
        list_graph_feature = []

        for line in lines[1:]:
            if line:
                line_list = line.split('\t')
                i_smiles1 = line_list[position_smiles1]
                i_smiles2 = line_list[position_smiles2]
                i_label = int(line_list[position_class])
                print(i_smiles1, i_smiles2, i_label)
                try:
                    i_graph = self.createGraph(i_smiles1, i_smiles2, i_label)
                    list_graph_feature.append(i_graph)
                except:
                    print('wrong')
                    continue
        print(len(list_graph_feature))
        list_graph_feature_new = []
        for d in list_graph_feature:
            i = Data(x=torch.tensor(d.x),
                     edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                     edge_attr=torch.tensor(d.edge_feats),
                     y=torch.tensor(d.y, dtype=torch.float32))
            list_graph_feature_new.append(i)

        self.feature = list_graph_feature_new

    def readTest(self, path_test: str):
        print('read predict result...')
        with open(path_test, 'rb') as f:
            lines = f.read().decode(encoding='utf-8').split('\r\n')
        table_list = lines[0].split('\t')
        self.data_info.append(table_list + ['empty_seperator', 'smile_before', 'smile_after', 'predict', 'class'])
        position_smiles1 = table_list.index('smile_before')
        position_smiles2 = table_list.index('smile_after')
        list_graph_feature = []

        for line in lines[1:]:
            if line:
                line_list = line.split('\t')
                i_smiles1 = line_list[position_smiles1]
                i_smiles2 = line_list[position_smiles2]
                # print(i_smiles1, i_smiles2)
                try:
                    i_graph = self.createGraph(i_smiles1, i_smiles2, -1)  # 这里的-1表示测试集中未知的类别
                    list_graph_feature.append(i_graph)
                    self.data_info.append(line_list + ['', i_smiles1, i_smiles2])
                except:
                    # print('wrong')
                    continue
        print('read done, valid result number and prepare to predict:', len(list_graph_feature))
        list_graph_feature_new = []
        for d in list_graph_feature:
            i = Data(x=torch.tensor(d.x),
                     edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                     edge_attr=torch.tensor(d.edge_feats),
                     y=torch.tensor(d.y, dtype=torch.float32))
            list_graph_feature_new.append(i)

        self.feature = list_graph_feature_new

    def train(self, flag_fold=False):

        loader_Train = self.feature
        train_loader = DataLoader(loader_Train, batch_size=64, shuffle=True, drop_last=False)

        n_epoch = 200
        net = CCPGraph()
        optimizer = torch.optim.Adam(params=net.parameters(), lr=0.002, betas=(0.9, 0.98), eps=1e-9)
        # optimizer = torch.optim.SGD(params=net.parameters(), lr=0.005)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.00001)
        softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)

        def loss_function(real, pred_logit, sampleW=None):
            # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
            cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
            return cross_ent.mean()

        weight_for_1 = torch.tensor(1, dtype=torch.float32)

        for epoch in range(n_epoch):

            start = time.time()
            list_pred, list_real = [], []
            net.train()
            for (batch, data) in enumerate(train_loader):
                optimizer.zero_grad()
                output, y_pred, att, _ = net(data)
                loss = loss_function(data.y, output, sampleW=weight_for_1)
                # loss = softmax_cross_entropy(y_pred, data.y)
                # print(epoch, '--', batch, '--', loss)
                # print(y_pred, data.y)
                loss.backward()
                optimizer.step()
                # print(epoch, batch,loss)
                list_real.extend(list(data.y.detach().numpy()))
                list_pred.extend(list(y_pred.detach().numpy()))
            # scheduler.step(epoch)
            list_pred = np.array(list_pred)
            list_pred[list_pred <= 0.5] = 0
            list_pred[list_pred > 0.5] = 1
            list_real = np.array(list_real)
            matrix = confusion_matrix(list_real, list_pred)
            print(matrix)
            tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
            print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
            tmp_train_bacc = (tpr + tnr) / 2
            train_time = time.time()
            print(train_time - start)

        self.param_dic = 'save/gcn_net_params-' + str(tmp_train_bacc) + '.pth'
        torch.save(net.state_dict(), self.param_dic)

        return net

    def validation(self, fold=5):

        loader_Train, loader_Valid = train_test_split(self.feature, test_size=1 / fold)
        print(len(loader_Train), len(loader_Valid))
        train_loader = DataLoader(loader_Train, batch_size=8, shuffle=True, drop_last=True)
        valid_loader = DataLoader(loader_Valid, batch_size=8, shuffle=True, drop_last=False)
        list_train_loss, list_val_loss, list_train_acc, list_val_acc = [], [], [], []

        n_epoch = 300
        net = CCPGraph()
        optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        # optimizer = torch.optim.SGD(params=net.parameters(), lr=0.005)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30, min_lr=0.00001)
        softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)

        def loss_function(real, pred_logit, sampleW=None):
            # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
            cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
            return cross_ent.mean()

        weight_for_1 = torch.tensor(1, dtype=torch.float32)
        best_val_bacc = 0.

        for epoch in range(n_epoch):
            list_loss = []
            start = time.time()
            list_pred, list_real = [], []
            net.train()
            for (batch, data) in enumerate(train_loader):
                optimizer.zero_grad()
                output, y_pred, att, _ = net(data)
                loss = loss_function(data.y, output, sampleW=weight_for_1)
                # loss = softmax_cross_entropy(y_pred, data.y)
                # print(epoch, '--', batch, '--', loss)
                # print(y_pred, data.y)
                loss.backward()
                optimizer.step()
                # print(epoch, batch,loss)
                list_loss.append(loss.detach().numpy())

                list_real.extend(list(data.y.detach().numpy()))
                list_pred.extend(list(y_pred.detach().numpy()))
            # scheduler.step(epoch)
            list_pred = np.array(list_pred)
            list_pred[list_pred <= 0.5] = 0
            list_pred[list_pred > 0.5] = 1
            list_real = np.array(list_real)
            matrix = confusion_matrix(list_real, list_pred)
            print(matrix)
            tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
            print('epoch:', epoch, '    Train:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
            tmp_train_bacc = (tpr + tnr) / 2
            list_train_loss.append(float(np.mean(list_loss)))
            list_train_acc.append((tpr + tnr) / 2)
            train_time = time.time()
            print(train_time - start)

            list_pred, list_real = [], []
            net.eval()
            list_loss = []
            for (batch, data) in enumerate(valid_loader):
                optimizer.zero_grad()
                output, y_pred, att, _ = net(data)
                loss = softmax_cross_entropy(y_pred, data.y)
                list_loss.append(loss.detach().numpy())
                # print(epoch, '--', batch, '--', loss)
                # print(y_pred, data.y)
                # print(epoch, batch,loss)
                list_real.extend(list(data.y.detach().numpy()))
                list_pred.extend(list(y_pred.detach().numpy()))
            list_pred = np.array(list_pred)
            list_pred[list_pred <= 0.5] = 0
            list_pred[list_pred > 0.5] = 1
            list_real = np.array(list_real)
            matrix = confusion_matrix(list_real, list_pred)
            print(matrix)
            tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
            print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
            tmp_val_bacc = (tpr + tnr) / 2
            list_val_loss.append(float(np.mean(list_loss)))
            list_val_acc.append((tpr + tnr) / 2)
            valid_time = time.time()
            print(valid_time - train_time)

            if tmp_val_bacc > best_val_bacc:
                self.param_dic = 'save/gcn_net_params-' + str(tmp_val_bacc) + '.pth'
                torch.save(net.state_dict(), self.param_dic)
                best_val_bacc = tmp_val_bacc

        plt.plot(np.arange(n_epoch), list_train_loss, color='b', label='train loss')
        plt.plot(np.arange(n_epoch), list_val_loss, color='r', label='validation loss')
        plt.ylim(min(list_train_loss) - 0.04, 2.5)
        plt.legend()
        plt.show()
        # list_val_acc = [min(i - 0.04, 0.824) for i in list_val_acc]
        print(max(list_val_acc), max(list_train_acc))
        plt.plot(np.arange(n_epoch), list_train_acc, color='b', label='train bacc')
        plt.plot(np.arange(n_epoch), list_val_acc, color='r', label='validation bacc')
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

        return net

    def predict(self, ):

        print('load pretrained model and predict...')
        loader_Test = self.feature
        test_loader = DataLoader(loader_Test, batch_size=64, shuffle=False, drop_last=False)
        self.net.load_state_dict(torch.load(self.param_dic))
        list_y = []
        for (batch, data) in enumerate(test_loader):
            output, y_pred, att, _ = self.net(data)
            list_y.extend(list(y_pred.tolist()))
        list_class = ['True' if i > 0.5 else 'False' for i in list_y]
        date_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print('predict done.')
        print('write predict result to ', 'predict\\' + str(date_now) + '.txt')
        with open('predict\\' + str(date_now) + '.txt', 'w') as f:
            f.write('\t'.join(self.data_info[0]) + '\n')
            for i in range(1, len(self.data_info)):
                f.write('\t'.join(self.data_info[i]) + '\t' + str(list_y[i-1]) + '\t' + str(list_class[i-1]) + '\n')


class ModTree:

    def __init__(self, dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info):

        self.dic_aa2mod = dic_aa2mod
        self.dic_mod_info = dic_mod_info
        self.dic_mod_index_info = dic_modindex_info
        self.dic_reac_group2mod = dic_reac_group2mod
        self.dic_reac_group2modindex = dic_reac_group2modindex
        self.dic_mod2index = dic_mod2index
        self.dic_index2mod = dic_index2mod
        self.iteration = 3

    def func_find_bfs(self, mass, ppm, aa, aa_position=''):

        # first iteration
        list_mod = self.dic_aa2mod[aa]
        list_result = []
        list_index = []
        if aa_position == 'N-term':
            list_mod = [i for i in list_mod if i[3] != 'Protein-N-Term']
            list_n_term = self.dic_aa2mod['N-term']
            list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
            list_mod.extend(list_n_term)
        elif aa_position == 'C-term':
            list_mod = [i for i in list_mod if i[3] != 'Protein-C-Term']
            list_c_term = self.dic_aa2mod['C-term']
            list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
            list_mod.extend(list_c_term)
        elif aa_position == 'Protein-N-term':
            list_n_term = self.dic_aa2mod['N-term']
            list_mod.extend(list_n_term)
        elif aa_position == 'Protein-C-term':
            list_c_term = self.dic_aa2mod['C-term']
            list_mod.extend(list_c_term)
        list_index.append(np.array([self.dic_mod2index[i[0]] for i in list_mod], dtype=np.int16))
        list_group = [i[-1] for i in list_mod]
        matrix_mass_dev = np.array([i[1] for i in list_mod], dtype=np.float32)
        matrix_mass_dev = mass - matrix_mass_dev
        flag_mass_dev = np.abs(matrix_mass_dev) <= ppm
        list_result.append(list_index[0][flag_mass_dev])
        del flag_mass_dev

        n = 1

        while n < self.iteration:
            print(n)
            # tmp iteration
            # 当前每一轮的mass列表、group列表、修饰列表大小是相同的，每一轮根据group列表计算当前轮每一个修饰对应下一轮的可能连接修饰是什么，
            next_index = np.array([(j, ver_index) for ver_index, tmp_group in enumerate(list_group) for i in tmp_group for j in self.dic_reac_group2modindex[i]], dtype=np.int32)
            print(next_index.shape)
            matrix_mass_dev = [matrix_mass_dev[i[1]]-self.dic_mod_index_info[i[0]][1]for i in next_index]
            flag_mass_dev = np.abs(matrix_mass_dev) <= ppm
            iter_result = next_index[flag_mass_dev]

            list_index.append(next_index)
            list_result.append(iter_result)

            if n < self.iteration - 1:
                list_group = [self.dic_mod_index_info[i][-1] for i in next_index[:, 0]]
                print(len(list_group))
            n += 1
            # list_result.append(list_index[0][flag_mass_dev])
            # next_index = [[[j, ver_index] for i in tmp_group for j in self.dic_reac_group2modindex[i]] for ver_index, tmp_group in enumerate(list_group)]
            # next_index_mass = [[self.dic_mod_index_info[j[0]][1] for j in i] for i in next_index if i]
            # matrix_mass_dev = [ for i in next_index_mass]
            # matrix_mass_dev = [matrix_mass_dev[i] - 1 for j in  for i in range(matrix_mass_dev.shape[0])]
            # list_group = [self.dic_mod_index_info[i[0]][-1] for i in next_index]

        return list_result

    def func_find_all(self, list_info):

        for i in list_info:
            mass, ppm, aa, aa_position = i
            self.func_find(mass,ppm,aa,aa_position)

    def func_find(self, mass, ppm, aa, aa_position='', leaf_reac_group='', flag_group='', iteration=0):

        if -1 * ppm <= mass <= ppm:
            return [[mass]]
        if iteration >= 3:
            return []

        # update
        if iteration == 0:
            list_mod = self.dic_aa2mod[aa]
            if aa_position == 'N-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-N-Term']
                list_n_term = self.dic_aa2mod['N-term']
                list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
                list_mod.extend(list_n_term)
            elif aa_position == 'C-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-C-Term']
                list_c_term = self.dic_aa2mod['C-term']
                list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
                list_mod.extend(list_c_term)
            elif aa_position == 'Protein-N-term':
                list_n_term = self.dic_aa2mod['N-term']
                list_mod.extend(list_n_term)
            elif aa_position == 'Protein-C-term':
                list_c_term = self.dic_aa2mod['C-term']
                list_mod.extend(list_c_term)
        else:
            list_mod = set()
            [list_mod.update(self.dic_reac_group2mod[tmp_reac_group]) for tmp_reac_group in leaf_reac_group if tmp_reac_group in self.dic_reac_group2mod]
            # for tmp_reac_group in leaf_reac_group:
            #     if tmp_reac_group in self.dic_reac_group2mod:
            #         tmp_mod = self.dic_reac_group2mod[tmp_reac_group]
            #         list_mod.update(tmp_mod)
            list_mod = [self.dic_mod_info[i] for i in list_mod]

        iteration += 1
        res = []

        if iteration == 3:
            # print(iteration, len(list_mod))
            if flag_group == 'chemical':
                list_mod = [i for i in list_mod if i[2] != 'biological']
            list_mass = mass - np.array([i[1] for i in list_mod])
            list_index = abs(list_mass) <= ppm
            list_res = [[list_mod[i][0], list_mass[i]] for i in range(len(list_index)) if list_index[i]]
            res += list_res
        else:
            # print(iteration, len(list_mod))
            if flag_group == 'chemical':
                list_mod = [i for i in list_mod if i[2] != 'biological']
            list_mass = np.float64(mass) - np.array([i[1] for i in list_mod])
            for i, tmp_mod_info in enumerate(list_mod):
                # print(iteration, tmp_mod_info[0], len(list_mod))
                sub_res = self.func_find(list_mass[i], ppm, aa, '', tmp_mod_info[-1], tmp_mod_info[2], iteration)
                res += [[tmp_mod_info[0]] + i for i in sub_res]

        return res

    def func_find_old(self, mass, ppm, aa, aa_position='', leaf_reac_group='', flag_group='', iteration=0):

        # update
        if iteration == 0:
            list_mod = self.dic_aa2mod[aa]
            if aa_position == 'N-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-N-Term']
                list_n_term = self.dic_aa2mod['N-term']
                list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
                list_mod.extend(list_n_term)
            elif aa_position == 'C-term':
                list_mod = [i for i in list_mod if i[3] != 'Protein-C-Term']
                list_c_term = self.dic_aa2mod['C-term']
                list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
                list_mod.extend(list_c_term)
            elif aa_position == 'Protein-N-term':
                list_n_term = self.dic_aa2mod['N-term']
                list_mod.extend(list_n_term)
            elif aa_position == 'Protein-C-term':
                list_c_term = self.dic_aa2mod['C-term']
                list_mod.extend(list_c_term)
        else:
            list_mod = set()
            for tmp_reac_group in leaf_reac_group:
                if tmp_reac_group in self.dic_reac_group2mod:
                    tmp_mod = self.dic_reac_group2mod[tmp_reac_group]
                    list_mod.update(tmp_mod)
            list_mod = [self.dic_mod_info[i] for i in list_mod]

        iteration += 1
        res = []

        if iteration >= 3:
            return []

        print(iteration, len(list_mod))
        for i, tmp_mod_info in enumerate(list_mod):
            # print(iteration, tmp_mod_info[0], len(list_mod))
            new_flag_group = tmp_mod_info[2]
            if new_flag_group == 'biological' and flag_group == 'chemical':
                continue
            if -1 * ppm <= mass-tmp_mod_info[1] <= ppm:
                sub_res = [[]]
            else:
                sub_res = self.func_find_old(mass-tmp_mod_info[1], ppm, aa, '', tmp_mod_info[-1], new_flag_group, iteration)
            res += [[tmp_mod_info[0]] + i for i in sub_res]

        return res

    def func_score(self, list_mod_result: list, aa, aa_position=''):

        for i, mod_result in enumerate(list_mod_result):

            mod_first = mod_result[0]
            score_mod_first = sum([1 + int(i[3] != 'Anywhere') for i in self.dic_aa2mod[aa] if i[0] == mod_first])
            if aa_position == 'N-term':
                list_n_term = self.dic_aa2mod['N-term']
                list_n_term = [i for i in list_n_term if i[3] != 'Protein-N-Term']
                score_mod_first += len([1 for i in list_n_term if i[0] == mod_first])
            elif aa_position == 'C-term':
                list_c_term = self.dic_aa2mod['C-term']
                list_c_term = [i for i in list_c_term if i[3] != 'Protein-C-Term']
                score_mod_first += len([1 for i in list_c_term if i[0] == mod_first])
            elif aa_position == 'Protein-N-term':
                list_n_term = self.dic_aa2mod['N-term']
                score_mod_first += len([1 for i in list_n_term if i[0] == mod_first])
            elif aa_position == 'Protein-C-term':
                list_c_term = self.dic_aa2mod['C-term']
                score_mod_first += len([1 for i in list_c_term if i[0] == mod_first])
            list_mod_result[i] = [score_mod_first - abs(float(mod_result[-1]))] + mod_result[:-1]

        list_mod_result.sort(key=lambda x:x[0], reverse=True)

        return list_mod_result

    def func_score_GCN(self, list_mod_result: list):

        net = ModGCNScore()
        for i, mod_result in enumerate(list_mod_result):
            mod_list = mod_result[:-1]
            mod_score = 0.
            i_score = net.predict(mod_list)



def func_load_mod_rule(path_modif):


    data_known_mod = pd.read_excel(path_modif, sheet_name=0)
    data_reactive_group1 = pd.read_excel(path_modif, sheet_name=1)
    data_reactive_group3 = pd.read_excel(path_modif, sheet_name=2)

    matrix_known_mod = data_known_mod.values
    matrix_reactive_group1 = data_reactive_group1.values
    matrix_reactive_group3 = data_reactive_group3.values
    list_delta_MS, list_Description, list_aa, list_classification, list_group, list_position, list_reactive_group2 = [], [], [], [], [], [], []

    dic_aa2mod = {}
    dic_mod_info = {}
    dic_modindex_info = {}
    dic_reac_group2mod = {}
    dic_reac_group2modindex = {}
    dic_mod2index = {}
    dic_index2mod = {}

    for i in range(matrix_reactive_group1.shape[0]):
        i_reactive_group1, i_description, i_delta_mass, i_class, i_group = matrix_reactive_group1[i]
        i_delta_mass = np.float32(i_delta_mass)
        i_reactive_group2 = i_reactive_group1.split(',')
        for tmp_reactive_group in i_reactive_group2:
            if tmp_reactive_group not in dic_reac_group2mod:
                dic_reac_group2mod[tmp_reactive_group] = set()

            dic_reac_group2mod[tmp_reactive_group].add(i_description)

        if i_description not in dic_mod_info:
            dic_mod_info[i_description] = [i_description, i_delta_mass, i_group, set()]


    for i in range(matrix_known_mod.shape[0]):
        i_delta_mass, i_description, i_aa, _, i_class, i_group, _, i_position, i_reactive_group2 = matrix_known_mod[i]
        i_delta_mass = np.float32(i_delta_mass)
        if i_description not in dic_mod2index:
            index = len(dic_mod2index)
            dic_mod2index[i_description] = index
            dic_index2mod[index] = i_description
        if i_reactive_group2 == i_reactive_group2:
            i_reactive_group2 = i_reactive_group2.split(',')
            i_reactive_group2 = [j for j in i_reactive_group2 if j in dic_reac_group2mod]
        else:
            i_reactive_group2 = []
        if i_aa not in dic_aa2mod:
            dic_aa2mod[i_aa] = []
        dic_aa2mod[i_aa].append([i_description, i_delta_mass, i_group, i_position, i_reactive_group2])

    for tmp_reactive_group in dic_reac_group2mod:
        dic_reac_group2modindex[tmp_reactive_group] = set()
        for j in dic_reac_group2mod[tmp_reactive_group]:
            dic_reac_group2modindex[tmp_reactive_group].add(dic_mod2index[j])

    for i in range(matrix_reactive_group3.shape[0]):
        i_description, i_reactive_group3 = matrix_reactive_group3[i]
        if i_reactive_group3 == i_reactive_group3:
            i_reactive_group3 = i_reactive_group3.split(',')
        else:
            i_reactive_group3 = []
        if i_description not in dic_mod_info:
            print('warning! {:s} not in dic_mod_info'.format(i_description))
        else:
            for tmp_reactive_group in i_reactive_group3:
                if tmp_reactive_group in dic_reac_group2mod:
                    dic_mod_info[i_description][-1].add(tmp_reactive_group)

    for i in dic_mod_info:
        dic_modindex_info[dic_mod2index[i]] = dic_mod_info[i]

    return dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info

def read_delta_mass(path: str):

    with open(path, 'rb')as f:
        lines = f.read().decode(encoding='utf-8').split('\r\n')
    list_info = []
    for line in lines[1:]:
        if line:
            line_list = line.split('\t')
            list_info.append([float(line_list[0]), line_list[1], line_list[2]])

    return list_info


if __name__ == '__main__':

    process_num = 1
    path_modif = r'data\mod_rule_20231028.xlsx'
    dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info = func_load_mod_rule(path_modif)
    modTree = ModTree(dic_aa2mod, dic_mod_info, dic_reac_group2mod, dic_mod2index, dic_index2mod, dic_reac_group2modindex, dic_modindex_info)

    list_info = read_delta_mass(r'data\data.txt')
    path_out = r'data\data_search.txt.txt'
    f_w = open(path_out, 'w')


    print(list_info)
    for i in list_info:
        print(i)
        delta_mass, aa, aa_position = i
        delta_mass = float(delta_mass)
        ppm_value = Global_ppm_threshold / 1e6 * abs(delta_mass)
        time_start = time.time()
        list_mod_tree = modTree.func_find(delta_mass, ppm_value, aa, aa_position)
        print(list_mod_tree)
        list_mod_tree = modTree.func_score(list_mod_tree, aa, aa_position)
        print(list_mod_tree)
        time_end = time.time()
        print(time_end - time_start)
        print(len(list_mod_tree))
        f_w.write('\t'.join([str(j) for j in i]) + '\n')
        f_w.write('the number of result: ' + str(len(list_mod_tree)) + '\n')
        for i, i_mod_tree in enumerate(list_mod_tree):
            f_w.write('\t'.join([str(j) for j in i_mod_tree]) + '\n')
            print(i, i_mod_tree)

    # 生成反应前后SMILES
    path_smiles = input("please input path of SMILES info before and after of reaction")
    
    # 读取
    modScore = ModGCNScore()
    modScore.param_dic = r'./save/gcn_net_params-0.8900373052446785.pth'
    modScore.readTest(path_smiles)
    modScore.predict()
