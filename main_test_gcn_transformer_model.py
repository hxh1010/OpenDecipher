import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from model_Transformer import TransformerEncoder
from model_gcn_new import CCPGraph
from feature_graph import Mol2Graph
from model_merge_gcn_transformer import Model_Merge

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


class CustomDataset(Dataset):
    def __init__(self, tensor_features, tensor_labels, graph_features):
        """
        Args:
            tensor_features: A tensor containing your features, e.g., torch.tensor(train_x)
            tensor_labels: A tensor containing your labels, e.g., torch.tensor(train_y)
            graph_features: Your graph features, e.g., a list of Data objects from torch_geometric
        """
        self.tensor_features = tensor_features
        self.tensor_labels = tensor_labels
        self.graph_features = graph_features

    def __len__(self):
        # Assuming tensor_features and graph_features are of the same length
        return len(self.tensor_features)

    def __getitem__(self, idx):
        # Retrieve the tensor features and labels
        tensor_feature = self.tensor_features[idx]
        tensor_label = self.tensor_labels[idx]

        # Retrieve the graph feature for this index
        graph_feature = self.graph_features[idx]

        # Return them as a tuple or in whatever format suits your needs
        return tensor_feature, tensor_label, graph_feature

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
        print('read predict result...')
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
                # print(i_smiles1, i_smiles2, i_label)
                try:
                    i_graph = self.createGraph(i_smiles1, i_smiles2, i_label)
                    list_graph_feature.append(i_graph)
                except:
                    # print('wrong')
                    continue
        print('read done, valid result number and prepare to train:', len(list_graph_feature))
        list_graph_feature_new = []
        for d in list_graph_feature:
            i = Data(x=torch.tensor(d.x),
                     edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                     edge_attr=torch.tensor(d.edge_feats),
                     y=torch.tensor(d.y, dtype=torch.float32))
            list_graph_feature_new.append(i)

        self.feature = list_graph_feature_new

    def readTest(self, path_test: str):

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
                print(i_smiles1, i_smiles2)
                try:
                    i_graph = self.createGraph(i_smiles1, i_smiles2, -1)  # 这里的-1表示测试集中未知的类别
                    list_graph_feature.append(i_graph)
                    self.data_info.append(line_list + ['', i_smiles1, i_smiles2])
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

        n_epoch = 200
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

        loader_Test = self.feature
        test_loader = DataLoader(loader_Test, batch_size=64, shuffle=False, drop_last=False)
        self.net.load_state_dict(torch.load(self.param_dic))
        list_y = []
        for (batch, data) in enumerate(test_loader):
            output, y_pred, att, _ = self.net(data)
            list_y.extend(list(y_pred.tolist()))
        list_class = ['True' if i > 0.5 else 'False' for i in list_y]
        date_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        with open('predict\\' + str(date_now) + '.txt', 'w') as f:
            f.write('\t'.join(self.data_info[0]) + '\n')
            for i in range(1, len(self.data_info)):
                f.write('\t'.join(self.data_info[i]) + '\t' + str(list_y[i-1]) + '\t' + str(list_class[i-1]) + '\n')


class ModTransformerScore:

    def __init__(self):

        self.net = TransformerEncoder()
        self.param_dic = ''
        self.data_info = []
        self.feature_x = []
        self.feature_y = []
        self.pos_number = 0
        self.neg_number = 0
        self.dicC2I = {}
        self.embed = []

    def createFeature(self, i_smiles1, i_smiles2, i_label=''):

        mol1 = Chem.MolFromSmiles(i_smiles1)
        mol2 = Chem.MolFromSmiles(i_smiles2)

        x = i_smiles1 + '&' + i_smiles2
        y = i_label

        return x, y

    def char2indices(self, listStr, dicC2I):
        listIndices = [0] * 200
        charlist = listStr
        for i, c in enumerate(charlist):
            if c not in dicC2I:
                dicC2I[c] = len(dicC2I)
                listIndices[i] = dicC2I[c] + 1
            else:
                listIndices[i] = dicC2I[c] + 1
        return listIndices

    def readTrain(self, path_train: str):

        with open(path_train, 'rb') as f:
            lines = f.read().decode(encoding='utf-8').split('\r\n')
        table_list = lines[0].split('\t')

        position_smiles1 = table_list.index('smile_before')
        position_smiles2 = table_list.index('smile_after')
        position_class = table_list.index('class')
        list_x, list_y = [], []

        for line in lines[1:]:
            if line:
                line_list = line.split('\t')
                i_smiles1 = line_list[position_smiles1]
                i_smiles2 = line_list[position_smiles2]
                i_label = float(line_list[position_class])
                print(i_smiles1, i_smiles2, i_label)
                try:
                    i_x, i_y = self.createFeature(i_smiles1, i_smiles2, i_label)
                    list_x.append(self.char2indices(i_x, self.dicC2I))
                    list_y.append(i_y)
                except:
                    print('wrong')
                    continue
        self.pos_number = list_y.count(1)
        self.neg_number = list_y.count(0)

        self.feature_x = list_x
        self.feature_y = list_y

    def readTest(self, path_test: str):

        with open(path_test, 'rb') as f:
            lines = f.read().decode(encoding='utf-8').split('\r\n')
        table_list = lines[0].split('\t')

        position_smiles1 = table_list.index('smile_before')
        position_smiles2 = table_list.index('smile_after')

        list_x, list_y = [], []

        for line in lines[1:]:
            if line:
                line_list = line.split('\t')
                i_smiles1 = line_list[position_smiles1]
                i_smiles2 = line_list[position_smiles2]

                print(i_smiles1, i_smiles2)
                try:
                    i_x, i_y = self.createFeature(i_smiles1, i_smiles2, '')
                    list_x.append(self.char2indices(i_x, self.dicC2I))
                    list_y.append(-1)
                    self.data_info.append([i_smiles1, i_smiles2, '-1'])
                except:
                    print('wrong')
                    continue

        self.feature_x = list_x
        self.feature_y = list_y

    def train(self, flag_fold=False):

        list_x = self.feature_x
        list_y = self.feature_y
        dataset = torch.utils.data.TensorDataset(torch.tensor(list_x), torch.tensor(list_y))
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

        n_epoch = 200
        num_layers = 2
        d_model = 100
        num_heads = 1
        dff = 1024
        seq_size = 200
        dropout_rate = 0.
        learning_rate = 0.001

        pos_num, neg_num = self.pos_number, self.neg_number
        initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
        # weight_for_1 = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
        weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
        print(weight_for_1)
        net = TransformerEncoder(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                                 dropout_rate)

        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

        def loss_function(real, pred_logit, sampleW=None):
            # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
            cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
            return cross_ent.mean()

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

        bit_size = 1024  # circular fingerprint
        embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
        nn.init.uniform(embeds.weight[1:], -1, 1)
        embeds.weight.requires_grad = False
        embeds.weight[1:].requires_grad = True

        tmp_train_bacc, tmp_val_bacc = 0, 0
        tmp_best_bacc = 0
        for epoch in range(n_epoch):
            start = time.time()

            list_pred, list_real = [], []
            net.train()
            for (batch, (X, Y)) in enumerate(train_loader):
                optimizer.zero_grad()
                mask = ~(X != 0)
                X = embeds(X)
                pred_logit_, y_pred = net(X, mask)
                loss = loss_function(Y, pred_logit_, sampleW=weight_for_1)
                # print(epoch, '--', batch, '--', loss)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # print(epoch, batch,loss)
                list_real.extend(list(Y.detach().numpy()))
                list_pred.extend(list(y_pred.detach().numpy()))
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

            # list_pred, list_real = [], []
            # net.eval()
            # for (batch, (X, Y)) in enumerate(valid_tf):
            #     mask = ~(X != 0)
            #     X = embeds(X)
            #     pred_logit_, y_pred = net(X, mask)
            #     list_real.extend(list(Y.detach().numpy()))
            #     list_pred.extend(list(y_pred.detach().numpy()))
            # list_pred = np.array(list_pred)
            # list_pred[list_pred <= 0.5] = 0
            # list_pred[list_pred > 0.5] = 1
            # list_real = np.array(list_real)
            # matrix = confusion_matrix(list_real, list_pred)
            # print(matrix)
            # tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            # tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
            # print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
            # tmp_val_bacc = (tpr + tnr) / 2
            # valid_time = time.time()
            # print(valid_time - train_time)
            #
            # for (batch, (X, Y)) in enumerate(test_tf):
            #     mask = ~(X != 0)
            #     X = embeds(X)
            #     pred_logit_, y_pred = net(X, mask)
            #     print('test')
            #     print(pred_logit_)
            #     print(y_pred)
            # if (tmp_train_bacc + tmp_val_bacc) > tmp_best_bacc:
            #     tmp_best_bacc = tmp_train_bacc + tmp_val_bacc
            #     torch.save(embeds.state_dict(),
            #                'save/embed1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
            #                    tmp_val_bacc) + '.pth')
            #     torch.save(net.state_dict(), 'save/net1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
            #         tmp_val_bacc) + '.pth')
            #     tmp_train_bacc, tmp_val_bacc = 0, 0

        self.param_dic = 'save/Transformer_net_params-' + str(tmp_train_bacc) + '.pth'
        torch.save(net.state_dict(), self.param_dic)

        return net

    def validation(self, fold=5):

        list_x = self.feature_x
        list_y = self.feature_y
        train_x, valid_x, train_y, valid_y = train_test_split(list_x, list_y, test_size=1 / fold)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)

        valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_x), torch.tensor(valid_y))
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True, drop_last=False)

        n_epoch = 200
        num_layers = 2
        d_model = 100
        num_heads = 1
        dff = 1024
        seq_size = 200
        dropout_rate = 0.
        learning_rate = 0.001

        pos_num, neg_num = self.pos_number, self.neg_number
        print("pos_num, neg_num", pos_num, neg_num)
        initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
        # weight_for_1 = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
        weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
        print(weight_for_1)
        net = TransformerEncoder(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                                 dropout_rate)

        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

        def loss_function(real, pred_logit, sampleW=None):
            # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
            cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
            return cross_ent.mean()

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

        bit_size = 1024  # circular fingerprint
        self.embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
        nn.init.uniform(self.embeds.weight[1:], -1, 1)
        self.embeds.weight.requires_grad = False
        self.embeds.weight[1:].requires_grad = True

        tmp_train_bacc, tmp_val_bacc = 0, 0
        tmp_best_bacc = 0
        for epoch in range(n_epoch):
            start = time.time()

            list_pred, list_real = [], []
            net.train()
            for (batch, (X, Y)) in enumerate(train_loader):
                optimizer.zero_grad()
                mask = ~(X != 0)
                X = self.embeds(X)
                pred_logit_, y_pred = net(X, mask)
                loss = loss_function(Y, pred_logit_, sampleW=weight_for_1)
                # print(epoch, '--', batch, '--', loss)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # print(epoch, batch,loss)
                list_real.extend(list(Y.detach().numpy()))
                list_pred.extend(list(y_pred.detach().numpy()))
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

            list_pred, list_real = [], []
            net.eval()
            for (batch, (X, Y)) in enumerate(valid_loader):
                mask = ~(X != 0)
                X = self.embeds(X)
                pred_logit_, y_pred = net(X, mask)
                list_real.extend(list(Y.detach().numpy()))
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
            valid_time = time.time()
            print(valid_time - train_time)

            # for (batch, (X, Y)) in enumerate(test_tf):
            #     mask = ~(X != 0)
            #     X = self.embeds(X)
            #     pred_logit_, y_pred = net(X, mask)
            #     print('test')
            #     print(pred_logit_)
            #     print(y_pred)
            # if (tmp_train_bacc + tmp_val_bacc) > tmp_best_bacc:
            #     tmp_best_bacc = tmp_train_bacc + tmp_val_bacc
            #     torch.save(embeds.state_dict(),
            #                'save/embed1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
            #                    tmp_val_bacc) + '.pth')
            #     torch.save(net.state_dict(), 'save/net1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
            #         tmp_val_bacc) + '.pth')
            #     tmp_train_bacc, tmp_val_bacc = 0, 0

        self.param_dic = 'save/Transformer_net_params-' + str(tmp_train_bacc) + '.pth'
        torch.save(net.state_dict(), self.param_dic)

        return net

    def predict(self, ):

        num_layers = 2
        d_model = 100
        num_heads = 1
        dff = 1024
        seq_size = 200
        dropout_rate = 0.

        pos_num, neg_num = self.pos_number, self.neg_number
        print("pos_num, neg_num", pos_num, neg_num)
        initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
        self.net = TransformerEncoder(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                                      dropout_rate)

        list_x = self.feature_x
        list_y = self.feature_y
        dataset = torch.utils.data.TensorDataset(torch.tensor(list_x), torch.tensor(list_y))
        test_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        self.net.load_state_dict(torch.load(self.param_dic))
        list_y = []

        for (batch, (X, Y)) in enumerate(test_loader):
            mask = ~(X != 0)
            X = self.embeds(X)
            pred_logit_, y_pred = self.net(X, mask)
            list_y.extend(list(y_pred.tolist()))

        date_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        with open('predict\\' + str(date_now) + '.txt', 'w') as f:
            f.write('\t'.join(['smile_before', 'smile_after', 'class', 'predict']) + '\n')
            for i in range(len(self.data_info)):
                f.write('\t'.join(self.data_info[i]) + '\t' + str(list_y[i]) + '\n')

        return list_y


class ModGCN_TransformerScore:

    def __init__(self):

        self.net = TransformerEncoder()
        self.param_dic = ''
        self.param_embed_dic = ''
        self.data_info = []
        self.feature = []
        self.feature_x = []
        self.feature_y = []
        self.pos_number = 0
        self.neg_number = 0
        self.dicC2I = {}
        self.embed = []


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

    def createFeature(self, i_smiles1, i_smiles2, i_label=''):

        mol1 = Chem.MolFromSmiles(i_smiles1)
        mol2 = Chem.MolFromSmiles(i_smiles2)

        x = i_smiles1 + '&' + i_smiles2
        y = i_label

        return x, y

    def char2indices(self, listStr, dicC2I):
        listIndices = [0] * 200
        charlist = listStr
        for i, c in enumerate(charlist):
            if c not in dicC2I:
                dicC2I[c] = len(dicC2I)
                listIndices[i] = dicC2I[c] + 1
            else:
                listIndices[i] = dicC2I[c] + 1
        return listIndices

    def readTrain(self, path_train: str):
        print('read predict result...')
        with open(path_train, 'rb') as f:
            lines = f.read().decode(encoding='utf-8').split('\r\n')
        table_list = lines[0].split('\t')

        position_smiles1 = table_list.index('smile_before')
        position_smiles2 = table_list.index('smile_after')
        position_class = table_list.index('class')
        list_x, list_y = [], []
        list_graph_feature = []

        for line in lines[1:]:
            if line:
                line_list = line.split('\t')
                i_smiles1 = line_list[position_smiles1]
                i_smiles2 = line_list[position_smiles2]
                i_label = int(line_list[position_class])
                # print(i_smiles1, i_smiles2, i_label)
                try:
                    i_graph = self.createGraph(i_smiles1, i_smiles2, i_label)
                    list_graph_feature.append(i_graph)
                    i_x, i_y = self.createFeature(i_smiles1, i_smiles2, i_label)
                    list_x.append(self.char2indices(i_x, self.dicC2I))
                    list_y.append(i_y)
                except:
                    # print('wrong')
                    continue
        print('read done, valid result number and prepare to train:', len(list_graph_feature))
        list_graph_feature_new = []
        for d in list_graph_feature:
            i = Data(x=torch.tensor(d.x),
                     edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                     edge_attr=torch.tensor(d.edge_feats),
                     y=torch.tensor(d.y, dtype=torch.float32))
            list_graph_feature_new.append(i)

        self.feature = list_graph_feature_new

        self.pos_number = list_y.count(1)
        self.neg_number = list_y.count(0)

        self.feature_x = list_x
        self.feature_y = list_y

    def readTest(self, path_test: str):
        print('read predict result...')
        with open(path_test, 'rb') as f:
            lines = f.read().decode(encoding='utf-8').split('\r\n')
        table_list = lines[0].split('\t')
        self.data_info.append(table_list + ['empty_seperator', 'smile_before', 'smile_after', 'predict', 'class'])
        position_smiles1 = table_list.index('smile_before')
        position_smiles2 = table_list.index('smile_after')
        list_graph_feature = []
        list_x, list_y = [], []

        for line in lines[1:]:
            if line:
                line_list = line.split('\t')
                i_smiles1 = line_list[position_smiles1]
                i_smiles2 = line_list[position_smiles2]
                try:
                    i_graph = self.createGraph(i_smiles1, i_smiles2, -1)  # 这里的-1表示测试集中未知的类别
                    list_graph_feature.append(i_graph)
                    i_x, i_y = self.createFeature(i_smiles1, i_smiles2, '')
                    list_x.append(self.char2indices(i_x, self.dicC2I))
                    list_y.append(-1)
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
        self.feature_x = list_x
        self.feature_y = list_y

    def train(self, flag_fold=False):

        list_x = self.feature_x
        list_y = self.feature_y
        loader_Train = self.feature
        dataset = torch.utils.data.TensorDataset(torch.tensor(list_x), torch.tensor(list_y), loader_Train)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=False)

        n_epoch = 200
        num_layers = 2
        d_model = 100
        num_heads = 1
        dff = 1024
        seq_size = 200
        dropout_rate = 0.
        learning_rate = 0.001

        pos_num, neg_num = self.pos_number, self.neg_number
        initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
        # weight_for_1 = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
        weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
        print(weight_for_1)
        net = Model_Merge(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                                 dropout_rate)

        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

        def loss_function(real, pred_logit, sampleW=None):
            # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
            cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
            return cross_ent.mean()

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

        bit_size = 1024  # circular fingerprint
        embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
        nn.init.uniform(embeds.weight[1:], -1, 1)
        embeds.weight.requires_grad = False
        embeds.weight[1:].requires_grad = True

        tmp_train_bacc, tmp_val_bacc = 0, 0
        tmp_best_bacc = 0
        for epoch in range(n_epoch):
            start = time.time()

            list_pred, list_real = [], []
            net.train()
            for (batch, (X, Y, data)) in enumerate(train_loader):
                optimizer.zero_grad()
                mask = ~(X != 0)
                X = embeds(X)
                pred_logit_, y_pred = net(X, mask, data)
                loss = loss_function(Y, pred_logit_, sampleW=weight_for_1)
                # print(epoch, '--', batch, '--', loss)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # print(epoch, batch,loss)
                list_real.extend(list(Y.detach().numpy()))
                list_pred.extend(list(y_pred.detach().numpy()))
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

            # list_pred, list_real = [], []
            # net.eval()
            # for (batch, (X, Y)) in enumerate(valid_tf):
            #     mask = ~(X != 0)
            #     X = embeds(X)
            #     pred_logit_, y_pred = net(X, mask)
            #     list_real.extend(list(Y.detach().numpy()))
            #     list_pred.extend(list(y_pred.detach().numpy()))
            # list_pred = np.array(list_pred)
            # list_pred[list_pred <= 0.5] = 0
            # list_pred[list_pred > 0.5] = 1
            # list_real = np.array(list_real)
            # matrix = confusion_matrix(list_real, list_pred)
            # print(matrix)
            # tnr = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            # tpr = matrix[1][1] / (matrix[1][1] + matrix[1][0])
            # print('epoch:', epoch, '    Valid:    TPR', tpr, 'TNR', tnr, 'bacc', (tpr + tnr) / 2)
            # tmp_val_bacc = (tpr + tnr) / 2
            # valid_time = time.time()
            # print(valid_time - train_time)
            #
            # for (batch, (X, Y)) in enumerate(test_tf):
            #     mask = ~(X != 0)
            #     X = embeds(X)
            #     pred_logit_, y_pred = net(X, mask)
            #     print('test')
            #     print(pred_logit_)
            #     print(y_pred)
            # if (tmp_train_bacc + tmp_val_bacc) > tmp_best_bacc:
            #     tmp_best_bacc = tmp_train_bacc + tmp_val_bacc
            #     torch.save(embeds.state_dict(),
            #                'save/embed1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
            #                    tmp_val_bacc) + '.pth')
            #     torch.save(net.state_dict(), 'save/net1_params-' + str(epoch) + '-' + str(tmp_train_bacc) + '-' + str(
            #         tmp_val_bacc) + '.pth')
            #     tmp_train_bacc, tmp_val_bacc = 0, 0

        self.param_dic = 'save/Transformer_net_params-' + str(tmp_train_bacc) + '.pth'
        torch.save(net.state_dict(), self.param_dic)

        return net


    def validation(self, fold=5):

        list_x = self.feature_x
        list_y = self.feature_y
        list_data = self.feature
        train_x, valid_x, train_y, valid_y, train_data, valid_data = train_test_split(list_x, list_y, list_data, test_size=1 / fold)
        list_train_loss, list_val_loss, list_train_acc, list_val_acc = [], [], [], []

        train_dataset = CustomDataset(torch.tensor(train_x), torch.tensor(train_y), train_data)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)

        valid_dataset = CustomDataset   (torch.tensor(valid_x), torch.tensor(valid_y), valid_data)
        valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, drop_last=False)

        n_epoch = 200
        num_layers = 2
        d_model = 100
        num_heads = 1
        dff = 1024
        seq_size = 200
        dropout_rate = 0.
        learning_rate = 0.001

        pos_num, neg_num = self.pos_number, self.neg_number
        initial_bias = float(np.log2(pos_num / neg_num) * 3 + 0.5)
        # weight_for_1 = torch.tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=torch.float32)
        # weight_for_1 = torch.tensor(neg_num / pos_num, dtype=torch.float32)
        weight_for_1 = torch.tensor(1, dtype=torch.float32)
        best_val_bacc = 0.

        net = Model_Merge(num_layers, d_model, num_heads, dff, initial_bias, seq_size,
                          dropout_rate)

        optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        softmax_cross_entropy = torch.nn.BCELoss(reduction='mean', reduce=True)

        def loss_function(real, pred_logit, sampleW=None):
            # cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real, pos_weight=sampleW)
            cross_ent = F.binary_cross_entropy_with_logits(pred_logit, real)
            return cross_ent.mean()

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)

        bit_size = 1024  # circular fingerprint
        embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
        nn.init.uniform(embeds.weight[1:], -1, 1)
        embeds.weight.requires_grad = False
        embeds.weight[1:].requires_grad = True

        tmp_train_bacc, tmp_val_bacc = 0, 0
        tmp_best_bacc = 0
        for epoch in range(n_epoch):
            start = time.time()
            list_loss = []
            list_pred, list_real = [], []
            net.train()
            for (batch, (X, Y, data)) in enumerate(train_loader):
                optimizer.zero_grad()
                mask = ~(X != 0)
                X = embeds(X)
                pred_logit_, y_pred = net(X, mask, data)
                loss = loss_function(Y.float(), pred_logit_, sampleW=weight_for_1)
                # loss = loss_function(Y.float(), pred_logit_)
                # print(epoch, '--', batch, '--', loss)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # print(epoch, batch,loss)
                list_loss.append(loss.detach().numpy())
                list_real.extend(list(Y.detach().numpy()))
                list_pred.extend(list(y_pred.detach().numpy()))
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
            for (batch, (X, Y, data)) in enumerate(valid_loader):
                optimizer.zero_grad()
                mask = ~(X != 0)
                X = embeds(X)
                pred_logit_, y_pred = net(X, mask, data)
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
                self.param_gcn_dic = 'save//' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + 'gcn_transformer_net_params-' + str(tmp_val_bacc) + '.pth'
                self.param_embed_dic = 'save//' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + 'embed_params-' + str(tmp_val_bacc) + '.pth'
                torch.save(net.state_dict(), self.param_gcn_dic)
                torch.save(embeds.state_dict(), self.param_embed_dic)

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

        self.param_dic = r'save/2024-06-13-13-41-03gcn_transformer_net_params-0.9016431924882629.pth'
        self.param_embed_dic = r'save/2024-06-13-13-41-03embed_params-0.9016431924882629.pth'

        num_layers = 1
        d_model = 100
        num_heads = 1
        dff = 1024
        seq_size = 200
        dropout_rate = 0.

        net = Model_Merge(num_layers, d_model, num_heads, dff, 0, seq_size,
                                      dropout_rate)

        list_x = self.feature_x
        list_y = self.feature_y
        list_data = self.feature

        test_dataset = CustomDataset(torch.tensor(list_x), torch.tensor(list_y), list_data)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)

        net.load_state_dict(torch.load(self.param_dic))
        bit_size = 1024  # circular fingerprint
        embeds = nn.Embedding(bit_size, d_model, padding_idx=0)
        nn.init.uniform(embeds.weight[1:], -1, 1)
        embeds.weight.requires_grad = False
        embeds.weight[1:].requires_grad = True
        embeds.load_state_dict(torch.load(self.param_embed_dic))
        list_y = []

        for (batch, (X, Y, data)) in enumerate(test_loader):
            mask = ~(X != 0)
            X = embeds(X)
            pred_logit_, y_pred = net(X, mask, data)
            list_y.extend(list(y_pred.tolist()))
        list_class = ['True' if i > 0.5 else 'False' for i in list_y]
        date_now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        with open('predict\\' + str(date_now) + '.txt', 'w') as f:
            f.write('\t'.join(self.data_info[0]) + '\n')
            for i in range(1, len(self.data_info)):
                f.write('\t'.join(self.data_info[i]) + '\t' + str(list_y[i - 1]) + '\t' + str(list_class[i - 1]) + '\n')

        return list_y


if __name__ == '__main__':

    path_test = r'data\test_data_20240125.txt'

    # 图卷积网络结合CNN+Transformer模型
    modScore = ModGCN_TransformerScore()
    # 和GCN不同，该模型需要输入两个模型参数文件，在训练过程中会自动生成gcn_transformer和embed为名称的两个文件，选择准确率高的直接复制文件名称即可
    modScore.param_dic = r'./save/2024-06-12-23-38-09gcn_transformer_net_params-0.8975743348982785.pth'
    modScore.param_embed_dic = r'./save/2024-06-12-23-38-09embed_params-0.8975743348982785.pth'
    modScore.readTest(path_test)
    modScore.predict()
