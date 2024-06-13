import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import math 

seed = 5
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.1):

        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, num_heads * self.depth, bias=False)
        self.wk = nn.Linear(d_model, num_heads * self.depth, bias=False)
        self.wv = nn.Linear(d_model, num_heads * self.depth, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.num_heads * self.depth, self.d_model, bias=False)

    def forward(self, v, k, q, mask=None):

        # v shape:[batch_size, len_seq, d_model
        batch_size = v.shape[0]
        q = self.wq(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, 1, 1)
            matmul_qk = matmul_qk.masked_fill(mask == 1, float("-inf"))

        attn = nn.Softmax(dim=-1)(matmul_qk)
        # attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.num_heads * self.depth)

        outputs = self.fc(context)

        return outputs


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff, bias=True)
        self.w_2 = nn.Linear(d_ff, d_model, bias=True)

    def init_parameter(self):

        torch.nn.init.xavier_normal_(self.w_1.bias, gain=1.0)
        torch.nn.init.xavier_normal_(self.w_2.bias, gain=1.0)

    def forward(self, x):

        output = self.w_2(F.relu(self.w_1(x)))
        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, dff)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, mask):

        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_out = out1
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1 + ffn_out)

        return out2


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers=2, d_model=100, num_heads=1, dff=200, output_bias=0, seq_len=200, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.seq_size = seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.output_bias = output_bias
        self.enc_layers = nn.ModuleList([])
        self.enc_layers.extend([
            TransformerEncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)
        self.semi_final = nn.Linear(d_model, 1, bias=False)

        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=5, stride=1)
        self.final_layer = nn.Linear(in_features=self.seq_size, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def init_parameter(self):
        torch.nn.init.constant_(self.final_layer.bias.data, val=self.output_bias)

    def forward(self, x, mask):

        x = x.reshape([-1, self.seq_size, self.d_model]).permute(0,2,1)
        x = self.conv1(x).permute(0,2,1)
        x = x.reshape([-1, self.seq_size-5+1, self.d_model])

        x = F.pad(x, pad=(0, 0, 0, 5-1, 0, 0), mode="constant", value=0)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        x = self.dropout(x)
        out = self.semi_final(x)
        out = out.reshape([-1, self.seq_size])
        dim = out
        out = self.final_layer(out)
        out = torch.squeeze(out, dim=1)

        return out, self.sigmoid(out)


class TransformerEncoderFeature(nn.Module):

    def __init__(self, num_layers=2, d_model=100, num_heads=1, dff=200, output_bias=0, seq_len=200, rate=0.1):
        super(TransformerEncoderFeature, self).__init__()
        self.seq_size = seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.output_bias = output_bias
        self.enc_layers = nn.ModuleList([])
        self.enc_layers.extend([
            TransformerEncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)])
        self.dropout = nn.Dropout(rate)
        self.semi_final = nn.Linear(d_model, 1, bias=False)

        self.conv1 = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=5, stride=1)
        self.final_layer = nn.Linear(in_features=self.seq_size, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def init_parameter(self):
        torch.nn.init.constant_(self.final_layer.bias.data, val=self.output_bias)

    def forward(self, x, mask):

        x = x.reshape([-1, self.seq_size, self.d_model]).permute(0,2,1)
        x = self.conv1(x).permute(0,2,1)
        x = x.reshape([-1, self.seq_size-5+1, self.d_model])

        x = F.pad(x, pad=(0, 0, 0, 5-1, 0, 0), mode="constant", value=0)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        x = self.dropout(x)
        out = self.semi_final(x)
        out = out.reshape([-1, self.seq_size])
        feature = out / 1e6

        return feature
