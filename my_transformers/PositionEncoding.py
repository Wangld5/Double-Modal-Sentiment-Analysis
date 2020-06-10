import torch
import torch.nn as nn
import numpy as np


class PositionEmbedding(nn.Module):
    """docstring for PositionEmbedding"""

    def __init__(self, d_model, max_len=1054):
        super(PositionEmbedding, self).__init__()
        self.register_buffer('pe', self.get_position_encoder(d_model, max_len))
        
    def get_position_encoder(self, d_model, max_len):
        def get_position(position):
            return [position / np.power(10000, 2*(i//2)/d_model) for i in range(d_model)]
        pe = np.array([get_position(pos_i) for pos_i in range(max_len)])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        return torch.FloatTensor(pe).unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__(4, d_model, padding_idx=0)

class BertEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1054):
        super(BertEmbedding, self).__init__()
        self.position_embedding = PositionEmbedding(d_model, max_len)
        self.segment_embedding = SegmentEmbedding(d_model)
    
    def forward(self, x):
        return x + self.position_embedding(x)


if __name__ == '__main__':
    test_arange = torch.arange(0, 512, 2)
    print(test_arange.shape)
