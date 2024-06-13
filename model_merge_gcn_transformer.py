import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import math
from model_Transformer import TransformerEncoderFeature
from model_gcn_new import CCPGraphFeature

seed = 5
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子

class Model_Merge(torch.nn.Module):

    def __init__(self, num_layers=1, d_model=100, num_heads=1, dff=200, output_bias=0, seq_len=200, rate=0.1):
        super().__init__()
        self.model_transformer = TransformerEncoderFeature(num_layers, d_model, num_heads, dff, output_bias, seq_len, rate)
        self.model_gcn = CCPGraphFeature()
        self.bn0 = nn.BatchNorm1d(400)
        self.lin1 = nn.Linear(200, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dp1 = nn.Dropout(0.1)
        self.final_layer = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()

    def init_parameter(self):
        torch.nn.init.constant_(self.final_layer.bias.data, val=2)
        self.final_layer.bias.requires_grad = False

    def forward(self, data_x, data_mask, data):

        feature_tranformer = self.model_transformer(data_x, data_mask)
        feature_graph = self.model_gcn(data)
        feature = feature_graph + feature_tranformer
        # feature = torch.cat([feature_tranformer, feature_graph], dim=1)

        # x = self.bn0(feature)
        # x = self.lin1(feature)
        # x = self.dp1(x)
        x = self.final_layer(feature)
        x = torch.squeeze(x, dim=1)

        return x, self.sigmoid(x)