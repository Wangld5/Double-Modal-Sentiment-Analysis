import torch
import torch.nn as nn
from my_transformers.transformer import Encoder
from my_transformers.PositionEncoding import PositionEmbedding
import os
import pdb

def padding_mask(y, x):
    seq_len = x.shape[1]
    x_sum = torch.sum(x, dim=2)
    pad_mask = x_sum.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, y.shape[1], seq_len).unsqueeze(1)
    return pad_mask

class transformer(nn.Module):
    def __init__(self, batch_size, num_layers, h, d_model, ffw_dim, dropout=0.1):
        super(transformer, self).__init__()
        self.d_model = d_model

        self.cls_sep_emb = nn.Embedding(2, d_model)
        self.enc = Encoder(num_layers, h, d_model, ffw_dim, dropout=dropout)
        self.output_linear = nn.Linear(d_model, 1, bias=True)

        self.Audio_conv1 = nn.Conv1d(74, d_model, kernel_size=3, stride=2, padding=0, bias=False)
        self.Vision_conv1 = nn.Conv1d(35, d_model, kernel_size=3, stride=2, padding=0, bias=False)
        self.Text_conv1 = nn.Conv1d(300, d_model, kernel_size=3, stride=2,  padding=0, bias=False)
        # MUTAN
        self.liner_T = nn.Linear(d_model, d_model)
        self.liner_V = nn.Linear(d_model, d_model)
        
        # self.position_embedding = PositionEmbedding(self.d_model)
        self.lstm = nn.LSTM(d_model, ffw_dim, 2, batch_first=True, bidirectional=True, dropout=dropout)
        self.lstm_linear = nn.Linear(2*ffw_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(self.d_model)
    
    def forward(self, Text, Audio, Vision):
        """
        Text: batch_size x seq_len x dim
        """

        #1) through a 1 dim convolution change dimention
        # the form is batch_size x seq_len x d_model
        Audio = Audio.transpose(-2, -1)
        Text = Text.transpose(-2, -1)
        Vision = Vision.transpose(-2, -1)

        Audio = self.Audio_conv1(Audio)
        # Audio = self.batchNorm(Audio)
        Text = self.Text_conv1(Text)
        # Text = self.batchNorm(Text)
        Vision = self.Vision_conv1(Vision)
        # Vision = self.batchNorm(Vision)

        Audio = Audio.transpose(-2, -1)
        Text = Text.transpose(-2, -1)
        Vision = Vision.transpose(-2, -1)

        # inner production
        # production_AT = Audio.mean(dim=1).mm(Text.mean(dim=1).T).mean(dim=1)
        # production_AV = Audio.mean(dim=1).mm(Vision.mean(dim=1).T).mean(dim=1)
        production_TV = Text.mean(dim=1).mm(Vision.mean(dim=1).T).mean(dim=1)
        production_avg = production_TV
        # production_avg = torch.cat(production_avg, dim=0)
        production_avg = torch.mean(production_avg) 
        production_avg = self.sigmoid(production_avg)
        production_loss = 1-production_avg

        #2) concat CLS label and three modal data
        cls_repr = self.cls_sep_emb(torch.zeros(Text.shape[0], 1).long())
        sep_repr = self.cls_sep_emb(torch.ones(Text.shape[0], 1).long())
        input_data = torch.cat((cls_repr, Text, sep_repr, Vision, sep_repr), dim=1)

        #3) mask of the input_data
        mask_V = padding_mask(Text, Vision)
        mask_T = padding_mask(Vision, Text)

        #4) add position embedding
        # input_data += self.position_embedding(input_data)
        # input_data = self.layernorm(input_data)

        #5) put the data in our model
        Vision_alpha, attn_score = self.enc(Text, Vision, mask_V)
        Text_alpha, _ = self.enc(Vision, Text, mask_T)

        #6) get label and predict
        # size is batch_size x 1
        input_data = torch.cat((Vision_alpha, Text_alpha), 1)
        # input_data = []
        # Vision_beta = self.liner_V(Vision_alpha)
        # Text_beta = self.liner_T(Text_alpha)
        # for i in range(self.d_model):
        #     input_data.append(torch.bmm(Vision_beta, Text_beta.permute(0, 2, 1)))
        # input_data = torch.stack(input_data, 1)
        # input_data = input_data.sum(1)

        self.lstm.flatten_parameters()
        input_data, _ = self.lstm(input_data)
        input_data = self.lstm_linear(input_data)
        pred_label = torch.mean(input_data, 1)
        # pred_label = self.output_linear(input_data)
        
        return pred_label, attn_score, production_loss


if __name__ == "__main__":
    a = torch.randn(20, 5)
    b = torch.randn(20, 5)
    print(a.mm(b.T).shape)



