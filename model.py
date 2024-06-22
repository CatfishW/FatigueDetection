import torch.nn as nn
import torch
from transformer import TransformerEncoder
#GRU
from torch.nn import GRU
class FatigueRegressionModel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, feature_length=7,target_length=5,dropout=0.1,):
        super().__init__()
        self.conv1 = self._conv1d(feature_length, d_model, 3, 1, 1)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, dim_feedforward, num_layers, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, target_length)
        )
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.conv1(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.norm1(x)
        x = self.transformer_encoder(x)
        x = self.norm2(x)
        #x = self.linear(x)
        x = self.decoder(x)
        return x
    def _conv1d(self, dim1, dim2, kernel_size, stride, padding, dropout=0.1):
        return nn.Sequential(
            nn.Conv1d(dim1, dim2, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
class FatigueClassificationModel(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, feature_length=7,num_cls=5,dropout=0.1,seq_len=10,trick=False):
        super().__init__()
        self.conv1 = self._conv1d(feature_length, d_model, 3, 1, 1)
        if trick:
            self.text_conv = self._conv1d(768, d_model, 3, 1, 1)
            self.mixer = self._conv1d(d_model, d_model, 3, 1, 1)
            self.text_norm = nn.LayerNorm(d_model)
        #self.encoder = TransformerEncoder(d_model, nhead, dim_feedforward, num_layers, dropout)
        self.encoder = GRU(d_model, d_model, num_layers, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_cls)
        self.norm2 = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(seq_len)
    def forward(self, x,y=None):
        x = self.conv1(x.transpose(-1,-2)).transpose(-1,-2)
        x = self.norm1(x)
        if y is not None:
            y = y.repeat(x.size(0),1,1).transpose(-1,-2)
            y = self.text_conv(y).transpose(-1,-2)
            y = self.text_norm(y)
            x = torch.cat([x,y],dim=1)
            x = self.mixer(x.transpose(-1,-2)).transpose(-1,-2)
        #x = self.encoder(x)
        x, _ = self.encoder(x)
        x = self.norm2(x)
        x = self.pool(x.transpose(-1,-2)).transpose(-1,-2).squeeze(-1)
        x = self.classifier(x)

        return x
    def _conv1d(self, dim1, dim2, kernel_size, stride, padding, dropout=0.1):
        return nn.Sequential(
            nn.Conv1d(dim1, dim2, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
