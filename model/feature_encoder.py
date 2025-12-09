import torch
from torch import nn
import math


from .unireplknet import UniRepLKNetBlock, DilatedReparamBlock
from .layers import ConvBNReLU


class DiffEncoder(nn.Module):
    def __init__(self, num_layers=2, d_model=512, nhead=8, dropout=0.5):
        super().__init__()
        
        self.bfda = BDFA(d_model, d_model)

        """Cross Attention Guided Differentiation (CAGD) Module"""
        self.pe_feat = PositionalEncoding(d_model, max_len=16*16)
        self.blocks = nn.ModuleList([RCDFF(d_model, dropout)
                                     for _ in range(num_layers)])

    def forward(self, imga, imgb, featd):
        fa, fb, fd = self.bfda(imga, imgb, featd)

        # n, c, h, w = feature.shape  # n, 512, 8, 8 for model_stage = 4 resnet18
        # feature = feature.view(n, c, -1).permute(2, 0,1)  # shape to (64, n, 512)
        # feature = self.pe_feat(feature)

        for blk in self.blocks:
            fa, fb, fd = blk(fa, fb, fd)
            
        n, c, h, w = fd.shape
        fd = fd.view(n, c, -1).permute(2, 0, 1)
        fd = self.pe_feat(fd)
        return fd  # seq_len, n, d_model = 64, n, 512


class BDFA(nn.Module):
    def __init__(self, d_dim=512, d_model=512):
        super(BDFA, self).__init__()
        self.convd1 = ConvBNReLU(d_dim, 2* d_model, 3, 1, 1)

        self.convd2 = nn.Sequential(
            ConvBNReLU(2* d_model, d_model, 1, 1, 0),
            ConvBNReLU(d_model, d_model, 3, 1, 1)
        )
        self.caa = ChannelAttention(d_model)
        self.cab = ChannelAttention(d_model)

        self.embed_img = nn.Conv2d(d_model, d_model, 3, 1, 1, groups=d_model)

    def forward(self, a, b, d):
        fd = self.convd1(d)  # n, 2*d_model, h, w
        xc= torch.cat([a, b], dim=1)
        fd = fd + xc
        fd = self.convd2(fd)  # n, d_model, h, w

        fa = self.embed_img(a)
        fa = self.caa(fa)
        fb = self.embed_img(b)
        fb = self.cab(fb)

        fa = fa * fd
        fb = fb * fd
        return fa, fb, fd
    
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction_ratio,
                      kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction_ratio,
                      in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        avg_out = self.mlp(self.avg_pool(image))
        max_img = nn.functional.adaptive_max_pool2d(image,(1,1))
        
        max_out = self.mlp(max_img)
        return self.sigmoid(avg_out + max_out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CRFF(nn.Module):
    def __init__(self, d_model=512, dropout=0.5):
        super(CRFF, self).__init__()
        self.replk1 = UniRepLKNetBlock(d_model, 7)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(d_model)
        # self.linear1 = nn.Conv2d(d_model, dim_feedforward, 1)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Conv2d(dim_feedforward, d_model, 1)
        self.replk2 = UniRepLKNetBlock(d_model, 7)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, image, feature):
        y = self.replk1(image)
        y = feature * y
        out = self.norm1(image + self.dropout1(y))
        # linear_out = self.linear2(self.dropout(self.relu(self.linear1(out))))
        # out = self.norm2(out + self.dropout2(linear_out))
        out = self.norm2(self.replk2(out))
        return out

class RCDFF(nn.Module):
    def __init__(self, d_model=512, dropout=0.3):
        super(RCDFF, self).__init__()
        self.crffa = CRFF(d_model, dropout)
        self.crffb = CRFF(d_model, dropout)

        self.zip = nn.Conv2d(2*d_model, d_model, 1)
        self.replk = UniRepLKNetBlock(d_model, 7)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm2d(d_model)


    def forward(self, before, after, feature):
    
        residual = feature  # TODO whether to add residual
        before = self.crffa(before, feature)
        after = self.crffb(after, feature)
        diff = torch.concat([before, after], dim=1)  # seq_len, n, d_model*2
        diff = self.zip(diff)
        diff = self.replk(diff)
        diff = self.dropout(diff)
        diff = self.norm(diff) + residual
        return before, after, diff
