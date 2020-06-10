import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class PositionwiseFeedForward(nn.Module):
    """
    position wise feed forward copy from transformer
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.w2(self.relu(self.w1(x)))
        return output


class AddandNorm(nn.Module):
    """
    a layer between multiheadAttention layer and feed forward layer.
    And it also be use between output and feed forward layer.
    """

    def __init__(self, size, dropout=0.1):
        super(AddandNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, result_x):
        """
        Apply residual network before layer norm
        """
        return self.norm(x + self.dropout(result_x))


class SingleAttention(nn.Module):
    def __init__(self, dropout=0.1):
        """
        dropout: the param for attention
        """
        super(SingleAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        """
        query: batch_size x h x seq_len x dim
        key: the same as query
        value: the same as query
        """
        # print('CLS: ', query[0, 0, 0, :])
        # print('last: ', query[0, 0, -1, :])
        attention = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(query.size(-1))
        if mask is not None:
            attention = attention.masked_fill(mask, -1e9)
        p_attn = self.softmax(attention)
        # print('attention: ', p_attn[0, 0, 0, :])
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        h: number of head, default 4
        d_model: the dimension of input 
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h

        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.attention = SingleAttention(dropout=dropout)

        # init weight
        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.reset_parameters()

    def forward(self, query, key, value, mask=None):
        """
        query: batch_size x seq_len x dimension
        key: the same as query
        value: the same as query
        """
        batch_size = query.size(0)

        # 1. do linear projections, dimension->h x d_k
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)  # k: batch x seq_len x (num_head*dim_perhead)
        v = self.in_proj_v(value)
        key = k.view(batch_size, k.shape[1], self.h, self.d_k).permute(0, 2, 1, 3)    # batch x self.h x seq_len x dim_perhead
        query = q.view(batch_size, q.shape[1], self.h, self.d_k).permute(0, 2, 1, 3)   
        value = v.view(batch_size, v.shape[1], self.h, self.d_k).permute(0, 2, 1, 3)

        # 2. apply attention
        x, attn_score = self.attention(query, key, value, mask=mask)

        # 3. concat the result above
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.h * self.d_k)

        return x, attn_score

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.in_proj_weight)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        weight = weight[start:end, :]
        return F.linear(input, weight)
    
    def in_proj_q(self, query):
        return self._in_proj(query, end=self.d_model)
    
    def in_proj_k(self, key):
        return self._in_proj(key, start=self.d_model, end=2 * self.d_model)
    
    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.d_model)

class EncoderLayer(nn.Module):
    """
    one block of encoder
    """

    def __init__(self, h, d_model, ffw_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(h, d_model, dropout=dropout)
        self.addNorm = AddandNorm(d_model, dropout=dropout)
        self.feedforward = PositionwiseFeedForward(d_model, ffw_dim, dropout=dropout)
        self.addNorm2 = AddandNorm(d_model, dropout=dropout)

    def forward(self, x, y, attn_mask=None):
        context, attn_score = self.attention(x, y, y, attn_mask)
        x = self.addNorm(x, context)
        output = self.feedforward(x)
        output = self.addNorm2(x, output)
        return output, attn_score


class Encoder(nn.Module):
    def __init__(self, num_layers, h, d_model, ffw_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.h = h
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(self.h, d_model, ffw_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x, y, attn_mask=None):
        """
        x: batch_size x seq_len x d_model
        attn_mask: batch_size x 1 x 1 x seq_len
        """
        for enc in self.encoder_layers:
            x, attn_score = enc(x, y, attn_mask)

        return x, attn_score


if __name__ == "__main__":
    tup = ()
    for i in range(10):
        tup  = tup + (i, )
    print(tup)
