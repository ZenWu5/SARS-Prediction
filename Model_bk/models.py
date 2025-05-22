# -*- coding: utf-8 -*-
"""
models.py
存放各类深度学习模型结构，包括：
- 通用RNN/LSTM/GRU模型
- AttentionBiLSTM模型
- RCNN模型
- AttentionModel模型
- DaRnnModel模型
- TransformerModel模型
以及模型工厂函数get_model
"""
import torch
import torch.nn.functional as F
from torch import nn


# -----------------------------
# 通用RNN/LSTM/GRU 模型
# -----------------------------
class RnnModel(nn.Module):
    """
    通用RNN模型，支持RNN、LSTM和GRU三种单元
    用于分类或回归任务，输出最后时刻的隐藏状态经过全连接层后的分数
    """
    def __init__(self, input_dim, output_dim, hidden_size, dropout_p, cell_type='LSTM'):
        super(RnnModel, self).__init__()
        # 保存超参数
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type.upper()
        # 输入Dropout
        self.dropout = nn.Dropout(dropout_p)
        # 根据cell_type选择不同单元
        if self.cell_type == 'LSTM':
            self.encoder = nn.LSTM(input_dim, hidden_size, batch_first=True)
        elif self.cell_type == 'GRU':
            self.encoder = nn.GRU(input_dim, hidden_size, batch_first=True)
        elif self.cell_type == 'RNN':
            self.encoder = nn.RNN(input_dim, hidden_size, batch_first=True)
        else:
            raise ValueError(f"不支持的cell_type: {cell_type}")
        # 输出层
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, x, hidden_state=None):
        """
        前向计算
        x: [batch_size, seq_len, input_dim]
        hidden_state: 可选隐藏状态
        返回: scores [batch_size, output_dim], dummy_attn [batch_size, seq_len]
        """
        x = self.dropout(x)
        if hidden_state is None:
            batch_size = x.size(0)
            hidden_state = self.init_hidden(batch_size)
        outputs, _ = self.encoder(x, hidden_state)
        last_output = outputs[:, -1, :]
        scores = self.out(last_output)
        dummy_attn = torch.zeros(outputs.size(0), outputs.size(1), device=x.device)
        return scores, dummy_attn

    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        device = next(self.parameters()).device
        if self.cell_type == 'LSTM':
            h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
            return (h0, c0)
        else:
            return torch.zeros(1, batch_size, self.hidden_size, device=device)


# -----------------------------
# AttentionBiLSTM 模型
# -----------------------------
class AttentionBiLSTM(nn.Module):
    """
    双向LSTM + 加性注意力机制，用于回归任务
    在时间维度上计算注意力权重，对LSTM输出加权求和后预测数值
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(AttentionBiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers>1 else 0.0
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x, mask=None):
        """
        前向计算
        x: [batch, seq_len, input_dim]
        mask: [batch, seq_len] 布尔型，False表示填充位置
        返回: out [batch,1], weights [batch, seq_len]
        """
        h, _ = self.lstm(x)
        scores = self.attn(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        context = torch.einsum('bs,bsh->bh', weights, h)
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        return out, weights


# -----------------------------
# RCNN 模型
# -----------------------------
class RCNN(nn.Module):
    """
    RCNN模型：GRU编码 + 1D卷积 + 最大池化 + 全连接，用于回归
    """
    def __init__(self, input_dim, hidden_size=64, dropout=0.2):
        super(RCNN, self).__init__()
        # 双向GRU编码器
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True, bidirectional=True)
        # 一维卷积层，将特征维度从hidden_size*2映射到hidden_size
        self.conv = nn.Conv1d(hidden_size*2, hidden_size, kernel_size=3, padding=1)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 全连接层输出单值
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, _mask=None):
        """
        前向计算
        x: [batch, seq_len, input_dim]
        返回: out [batch,1], None (无注意力)
        """
        # GRU编码
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden*2]
        # 调整维度为Conv1d输入格式
        conv_in = gru_out.permute(0,2,1)  # [batch, features, seq_len]
        conv_out = self.conv(conv_in)     # [batch, hidden, seq_len]
        # 恢复到 [batch, seq_len, hidden]
        conv_out = conv_out.permute(0,2,1)
        # 最大池化获取序列维度上全局特征
        pooled, _ = torch.max(conv_out, dim=1)  # [batch, hidden]
        # Dropout + 全连接
        out = self.dropout(pooled)
        out = self.fc(out)
        return out, None


# -----------------------------
# AttentionModel 模型
# -----------------------------
class AttentionModel(nn.Module):
    """
    时间注意力模型：LSTM编码 + 隐藏态注意力
    """
    def __init__(self, seq_length, input_dim, output_dim, hidden_size, dropout_p):
        super(AttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.encoder = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, seq_length)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, x, hidden_state=None):
        x = self.dropout(x)
        batch_size = x.size(0)
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        outputs, (h, _) = self.encoder(x, hidden_state)
        attn_weights = F.softmax(self.attn(h.squeeze(0)), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)
        scores = self.out(context)
        return scores, attn_weights

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h0, c0)


# -----------------------------
# DaRnnModel 模型
# -----------------------------
class DaRnnModel(nn.Module):
    """
    双重注意力RNN：输入注意力 + 时间注意力
    """
    def __init__(self, seq_length, input_dim, output_dim, hidden_size, dropout_p):
        super(DaRnnModel, self).__init__()
        self.T = seq_length
        self.m = hidden_size
        self.n = input_dim
        self.dropout = nn.Dropout(dropout_p)
        self.encoder = nn.LSTM(self.n, self.m, batch_first=True)
        self.We = nn.Linear(2*self.m, self.T)
        self.Ue = nn.Linear(self.T, self.T)
        self.ve = nn.Linear(self.T, 1)
        self.Ud = nn.Linear(self.m, self.m)
        self.vd = nn.Linear(self.m, 1)
        self.out = nn.Linear(self.m, output_dim)

    def forward(self, x, hidden_state=None):
        x = self.dropout(x)
        batch_size = x.size(0)
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        h_seq = []
        for t in range(self.T):
            hc = torch.cat([hidden_state[0].transpose(0,1), hidden_state[1].transpose(0,1)], dim=2)
            e = self.ve(torch.tanh(self.We(hc)+self.Ue(x.transpose(1,2)))).squeeze(-1)
            alpha = F.softmax(e, dim=1)
            x_tilde = (alpha.unsqueeze(-1)*x[:,t,:]).unsqueeze(1)
            ht, hidden_state = self.encoder(x_tilde, hidden_state)
            h_seq.append(ht)
        h_cat = torch.cat(h_seq, dim=1)
        l = self.vd(torch.tanh(self.Ud(h_cat))).squeeze(-1)
        beta = F.softmax(l, dim=1)
        context = torch.bmm(beta.unsqueeze(1), h_cat).squeeze(1)
        logits = self.out(context)
        return logits, beta

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.m, device=device)
        c0 = torch.zeros(1, batch_size, self.m, device=device)
        return (h0, c0)


# -----------------------------
# Transformer 模型
# -----------------------------
class TransformerModel(nn.Module):
    """
    基于Transformer Encoder的模型，适用于回归或分类
    """
    def __init__(self, input_dim, output_dim, dropout_p=0.1, nhead=5, num_layers=2):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dropout=dropout_p, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, hidden_state=None):
        out = self.transformer(x)
        last = out[:, -1, :]
        scores = self.fc(last)
        return scores, None


# -----------------------------
# 模型工厂函数
# -----------------------------

def get_model(name, **kwargs):
    """
    根据名称获取对应模型实例
    参数:
      name: 'bilstm','rnn','lstm','gru','attention','rcnn','da-rnn','transformer'
      kwargs: 对应模型构造函数所需参数
    返回:
      模型实例
    """
    name = name.lower()
    if name == 'bilstm':
        return AttentionBiLSTM(**kwargs)
    if name in ('rnn','lstm','gru'):
        cell = name.upper() if name!='rnn' else 'RNN'
        return RnnModel(input_dim=kwargs.get('input_dim'),
                        output_dim=kwargs.get('output_dim'),
                        hidden_size=kwargs.get('hidden_size'),
                        dropout_p=kwargs.get('dropout_p'),
                        cell_type=cell)
    if name == 'attention':
        return AttentionModel(**kwargs)
    if name == 'rcnn':
        return RCNN(input_dim=kwargs.get('input_dim'),
                    hidden_size=kwargs.get('hidden_size'),
                    dropout=kwargs.get('dropout'))
    if name in ('da-rnn','dar nn'):
        return DaRnnModel(**kwargs)
    if name == 'transformer':
        return TransformerModel(**kwargs)
    raise ValueError(f"未知模型名称: {name}")

