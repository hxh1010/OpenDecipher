from functionExtractFeature import CFunctionExtractFeature, CFunctionExtractFeatureConvForFinetune
from functionExtractFingerPrint import CFunctionExtractFeatureDesc
from feature_graph import Mol2Graph
from rdkit.Chem import AllChem as Chem
import torch
import random
from torch_geometric.data import Data #, DataLoader
from torch_geometric.loader import DataLoader
import random
import torch
import numpy as np
import pickle


def tool_PreprocessDataForGCNFineture(list_graph_feature, list_label):

    graph_size = np.array([i_v[2].shape[0] for i_v in list_graph_feature])
    max_graph_size = max(graph_size)
    graph_vertices = []
    graph_adjacency = []

    FEATURE1_SUBGRAPH_SIZE = [i[0] for i in list_graph_feature]
    FEATURE2_GLOBAL_STATE = [i[1] for i in list_graph_feature]
    FEATURE3_V = [i[2] for i in list_graph_feature]
    FEATURE4_A = [i[3] for i in list_graph_feature]
    N_FEATURE = len(list_graph_feature)

    for i in range(len(list_graph_feature)):
        print(i)
        graph_vertices.append(np.pad(FEATURE3_V[i],
                                     pad_width=(
                                     (0, max_graph_size - FEATURE3_V[i].shape[0]), (0, 0)),
                                     mode='constant', constant_values=0.))
        graph_adjacency.append(np.pad(FEATURE4_A[i],
                                      pad_width=(
                                      (0, max_graph_size - FEATURE4_A[i].shape[0]), (0, 0),
                                      (0, max_graph_size - FEATURE4_A[i].shape[0])),
                                      mode='constant', constant_values=0))

    MATRIX_V = np.stack(graph_vertices, axis=0)
    MATRIX_A = np.stack(graph_adjacency, axis=0)

    MATRIX_SG = np.array(FEATURE1_SUBGRAPH_SIZE)
    MATRIX_G = np.array(FEATURE2_GLOBAL_STATE)
    VECTOR_LABEL = np.array(list_label)

    # with open(r'F:\PycharmProjects\UltraTCM\python_script\CoAggregators\out\out1_FeatureFinetune_preprocess.pkl',
    #           'wb') as f:
    #
    #     pickle.dump([MATRIX_V,
    #                  MATRIX_A,
    #                  MATRIX_SG,
    #                  MATRIX_G,
    #                  MATRIX_E,
    #                  VECTOR_LABEL,
    #                  N_FEATURE],
    #                 f, protocol=4)

    return [MATRIX_V,MATRIX_A,MATRIX_SG,MATRIX_G,VECTOR_LABEL,N_FEATURE]

def tool_SaveFeature_multimodel(list_feature, name_save):

    # with open(r'save\feature_save_all\out_feature_embed.pkl',
    #           'wb') as f:
    #     pickle.dump(list_feature[0],
    #                 f, protocol=4)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_graph.pkl',
              'wb') as f:
        pickle.dump(list_feature[0],
                    f, protocol=4)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_str.pkl',
              'wb') as f:
        pickle.dump(list_feature[1],
                    f, protocol=4)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_fingerprint.pkl',
              'wb') as f:
        pickle.dump(list_feature[2],
                    f, protocol=4)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_label.pkl',
              'wb') as f:
        pickle.dump(list_feature[3],
                    f, protocol=4)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_dic.pkl',
              'wb') as f:
        pickle.dump(list_feature[4],
                    f, protocol=4)

def tool_LoadFeature_multimodel(name_save):

    # with open(r'save\feature_save_all\out_feature_embed.pkl',
    #           'rb') as f:
    #     feature_embed = pickle.load(f)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_graph.pkl',
              'rb') as f:
        feature_graph = pickle.load(f)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_str.pkl',
              'rb') as f:
        feature_str = pickle.load(f)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_fingerprint.pkl',
              'rb') as f:
        feature_fingerprint = pickle.load(f)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_label.pkl',
              'rb') as f:
        feature_label = pickle.load(f)

    with open(r'save\feature_save_all\\'+ name_save + 'out_feature_dic.pkl',
              'rb') as f:
        feature_dic = pickle.load(f)

    return feature_label, [feature_graph, feature_str, feature_fingerprint], feature_dic

def func_get_feature_multimodel(list_smiles, name_save='',input_dic={}):

    class DataFeat(object):
        def __init__(self, **kwargs):
            for k in kwargs:
                self.__dict__[k] = kwargs[k]

    functionExtractEmbed = CFunctionExtractFeatureConvForFinetune()
    functionExtractFinger = CFunctionExtractFeatureDesc()
    set_smiles = set()
    for i in range(len(list_smiles)):
        i_smiles1, i_smiles2, _ = list_smiles[i]
        set_smiles.add(i_smiles1)
        set_smiles.add(i_smiles2)
    functionExtractEmbed.calculateEmbedding(set_smiles)

    list_graph_feature = []
    list_fingerprint = []
    list_label = []

    dicC2I_ = input_dic
    list_X = []
    list_smiles_combine = []

    count_invalid = 0
    cntTooLong = 0
    count_valid = 0
    for i in range(len(list_smiles)):
        i_smiles1, i_smiles2, i_label = list_smiles[i]
        i_smiles_combine = i_smiles1 + '&' + i_smiles2

        if i % 10 == 0:
            print('process:', i, 'invalid:', count_invalid)
        if len(i_smiles_combine) <= 200:
            try:

                mol1 = Chem.MolFromSmiles(i_smiles1)
                mol2 = Chem.MolFromSmiles(i_smiles2)

                g1 = Mol2Graph(mol1)
                g2 = Mol2Graph(mol2)
                x = np.concatenate([g1.x, g2.x], axis=0)
                edge_feats = np.concatenate([g1.edge_feats, g2.edge_feats], axis=0)
                e_idx2 = g2.edge_idx + g1.node_num
                edge_index = np.concatenate([g1.edge_idx, e_idx2], axis=0)
                tmp_graph = DataFeat(x=x, edge_feats=edge_feats, edge_index=edge_index, y=np.array([i_label], dtype=np.float))

                tmp_embedding = functionExtractEmbed.extract(i_smiles1, i_smiles2)
                tmp_embedding = list(tmp_embedding.squeeze().detach().numpy())
                tmp_fingerprint = functionExtractFinger.extract(i_smiles1, i_smiles2)
                tmp_fingerprint.extend(tmp_embedding)

                list_graph_feature.append(tmp_graph)
                list_fingerprint.append(tmp_fingerprint)
                list_label.append(i_label)

                if len(i_smiles_combine) > 200:
                    cntTooLong += 1
                    count_invalid += 1
                    continue
                list_smiles_combine.append(i_smiles_combine)
                list_X.append(tool_char2indices(i_smiles_combine, dicC2I_))  # length can vary

                count_valid += 1
            except:
                info = r"can not load molecular pair {:s}: ".format(i_smiles1 + '\t' + i_smiles2)
                count_invalid += 1
                print(info)
        else:
            count_invalid += 1
    print('提取到的有效特征的分子对数目：', count_valid)
    print(dicC2I_)
    list_graph_feature_new = []
    for d in list_graph_feature:
        i = Data(x=torch.tensor(d.x),
                 edge_index=torch.tensor(d.edge_index.T, dtype=torch.long),
                 edge_attr=torch.tensor(d.edge_feats),
                 y=torch.tensor(d.y, dtype=torch.float32))
        list_graph_feature_new.append(i)

    tool_SaveFeature_multimodel([list_graph_feature_new, list_X, list_fingerprint, list_label, dicC2I_], name_save)
    list_label, [list_graph_feature, list_X, list_fingerprint], dicC2I_ = tool_LoadFeature_multimodel(name_save)

    return list_label, [list_graph_feature, list_X, list_fingerprint], dicC2I_

def func_get_feature_multimodel_old(list_smiles):

    functionExtractGraph = CFunctionExtractFeature()
    functionExtractEmbed = CFunctionExtractFeatureConvForFinetune()
    functionExtractFinger = CFunctionExtractFeatureDesc()
    set_smiles = set()
    for i in range(len(list_smiles)):
        i_smiles1, i_smiles2, _ = list_smiles[i]
        set_smiles.add(i_smiles1)
        set_smiles.add(i_smiles2)
    functionExtractEmbed.calculateEmbedding(set_smiles)

    list_embedding = []
    list_graph_feature = []
    list_fingerprint = []
    list_label = []

    dicC2I_ = {}
    list_X = []
    list_smiles_combine = []

    count_invalid = 0
    cntTooLong = 0
    count_valid = 0
    for i in range(len(list_smiles)):
        i_smiles1, i_smiles2, i_label = list_smiles[i]
        i_smiles_combine = i_smiles1 + '&' + i_smiles2

        if i % 10 == 0:
            print('process:', i, 'invalid:', count_invalid)
        if len(i_smiles_combine) <= 200:
            try:
                tmp_dic_feature = functionExtractGraph.extractBaseSmile(i_smiles1, i_smiles2)
                tmp_embedding = functionExtractEmbed.extract(i_smiles1, i_smiles2)
                tmp_fingerprint = functionExtractFinger.extract(i_smiles1, i_smiles2)
                list_graph_feature.append([tmp_dic_feature['subgraph_size'], tmp_dic_feature['global_state'], tmp_dic_feature['V'], tmp_dic_feature['A']])
                list_embedding.append(tmp_embedding.squeeze().detach().numpy())
                list_fingerprint.append(tmp_fingerprint)
                list_label.append(i_label)

                if len(i_smiles_combine) > 200:
                    cntTooLong += 1
                    count_invalid += 1
                    continue
                list_smiles_combine.append(i_smiles_combine)
                list_X.append(tool_char2indices(i_smiles_combine, dicC2I_))  # length can vary

                count_valid += 1
            except:
                info = r"can not load molecular pair {:s}: ".format(i_smiles1 + '\t' + i_smiles2)
                count_invalid += 1
                print(info)
        else:
            count_invalid += 1
    print('提取到的有效特征的分子对数目：', count_valid)
    list_graph_feature = tool_PreprocessDataForGCNFineture(list_graph_feature, list_label)

    tool_SaveFeature_multimodel([list_embedding, list_graph_feature, list_X, list_fingerprint, list_label])
    list_label, [list_embedding, list_graph_feature, list_X, list_fingerprint] = tool_LoadFeature_multimodel()

    return list_label, [list_embedding, list_graph_feature, list_X, list_fingerprint]

def func_get_feature_embedding(list_smiles):

    functionExtractEmbed = CFunctionExtractFeatureConvForFinetune()
    set_smiles = set()
    for i in range(len(list_smiles)):
        i_smiles1, i_smiles2, _ = list_smiles[i]
        set_smiles.add(i_smiles1)
        set_smiles.add(i_smiles2)
    functionExtractEmbed.calculateEmbedding(set_smiles)

    count_invalid = 0
    list_embedding = []
    list_label = []
    count_valid = 0
    for i in range(len(list_smiles)):
        i_smiles1, i_smiles2, i_label = list_smiles[i]
        # print(i_smiles1 + '\t' + i_smiles2)
        if i % 100 == 0:
            print('process:', i, 'invalid:', count_invalid)
        try:
            tmp_embedding = functionExtractEmbed.extract(i_smiles1, i_smiles2)
            list_embedding.append(tmp_embedding.squeeze().detach().numpy())
            list_label.append(i_label)
            count_valid += 1
        except:
            info = r"can not load molecular pair {:s}: ".format(i_smiles1 + '\t' + i_smiles2)
            count_invalid += 1
            print(info)
    print('提取到的有效特征的分子对数目：', count_valid)

    return list_label, list_embedding

    # return train_loader, valid_loader

def func_get_feature_str(list_smiles):

    dicC2I_ = {}
    list_X = []
    list_smiles_combine = []
    count_invalid = 0
    cntTooLong = 0
    for i in range(len(list_smiles)):
        i_smiles1, i_smiles2, i_label = list_smiles[i]
        i_smiles_combine = i_smiles1 + '&' + i_smiles2
        # print(i_smiles1 + '\t' + i_smiles2)
        if i % 100 == 0:
            print('process:', i, 'invalid:', count_invalid)
        if len(i_smiles_combine) >= 200:
            cntTooLong += 1
            count_invalid += 1
            continue
        list_smiles_combine.append(i_smiles_combine)
        list_X.append(tool_char2indices(i_smiles_combine, dicC2I_))#length can vary

    dataset = torch.utils.data.TensorDataset(torch.tensor(list_X))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    return data_loader, list_smiles_combine

def tool_char2indices(listStr, dicC2I):
    listIndices = [0]*200
    charlist = listStr
    for i, c in enumerate(charlist):
        if c not in dicC2I:
            dicC2I[c] = len(dicC2I)
            listIndices[i] = dicC2I[c]+1
        else:
            listIndices[i] = dicC2I[c]+1
    return listIndices


