import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

from .layers import ConvBNReLU
from .dinoadaptor import DINOv3_ViT
from .feature_encoder import DiffEncoder
from .captiondecoder import TransDecoder


class Dino_DFFCC(nn.Module):
    def __init__(self, args, vocab_size):
        """args should contain keys as follow:
            model_stage: int
            proj_channel: int
            ft_layer: int
            encoder_n_layers: int
            vocab_size: int
            n_head: int
            dropout: float
            decoder_n_layers: int
        """
        super().__init__()
        
        d_model = 1024 
        self.extractor = DINOv3_ViT(d_model)

        """Difference Feature Encoder"""
        # self.encoder = DiffEncoder(num_layers=args.encoder_n_layers,
        #                            d_model=d_model,
        #                            nhead=args.n_heads,
        #                            dropout=args.dropout)
        self.encoder = DiffEncoder(num_layers=args.encoder_n_layers,
                                   d_model=d_model,
                                   nhead=args.n_heads,
                                   dropout=args.dropout)
        
        """Caption Decoder"""
        self.decoder = TransDecoder(feature_dim=d_model,
                                    vocab_size=vocab_size,
                                    n_head=args.n_heads,
                                    n_layers=args.decoder_n_layers,
                                    dropout=args.dropout)
        


    def forward(self, images, captions, cap_lens):
        feature = self.extractor(images)
        feature = self.encoder(images, feature)
        feature = self.decoder(feature, captions, cap_lens)
        return feature 
