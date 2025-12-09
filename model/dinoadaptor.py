# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial

from dinov3.hub.encoder import dinov3_vits16, dinov3_vitb16, dinov3_vitl16
from dinov3.hub.encoder import dinov3_convnext_tiny, dinov3_convnext_small, dinov3_convnext_base, dinov3_convnext_large
from .layers import ConvBNReLU

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DINOv3_ViT(nn.Module):
    def __init__(
        self,
        d_model,
        img_size=256,
        dinov3_model="dinov3_l",
    ):
        super().__init__()
        self.dino = DINOv3_MODEL_FACTORIES[dinov3_model](img_size=img_size, weights=pretrained_urls[dinov3_model])
        # Important: we freeze the dino
        self.dino.requires_grad_(False)

        self.img_size = img_size
        self.img_size_d16 = int(img_size // self.dino.patch_size)
        self.embed_dim = self.dino.embed_dim

        self.interaction_indexes = DINOv3_INTERACTION_INDEXES[dinov3_model]

        self.conv1 = ConvBNReLU(6, d_model, 3, 2, 1)
        self.down1 = nn.MaxPool2d(3, 2, 1)
        # self.conv2 = nn.Conv2d(d_model, d_model, 3,2,1, groups=d_model)
        self.conv2 = ConvBNReLU(d_model, d_model, 3, 2, 1)
        self.down2 = nn.MaxPool2d(3, 2, 1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:, 3:, :, :]
            x1 = x1[:, :3, :, :]

        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                y1lsit = self.dino.get_intermediate_layers(
                    x1, n=self.interaction_indexes, return_class_token=False
                )
                y2list = self.dino.get_intermediate_layers(
                    x2, n=self.interaction_indexes, return_class_token=False
                )
        
        y1 = y1lsit[-1]
        y2 = y2list[-1]
        # y = torch.concat([y1, y2], -1)

        b, l , c = y1.shape
        l = int(l**0.5)
        y1 = y1.reshape(b, l, l, c).permute(0, 3, 1, 2).contiguous()
        y2 = y2.reshape(b, l, l, c).permute(0, 3, 1, 2).contiguous()
        yc = torch.concat([x1, x2], 1)
        # y = y.reshape(b, l, l, c).permute(0, 3, 1, 2).contiguous()
        y = self.conv1(yc)
        y = self.down1(y)
        y = self.conv2(y)
        y = self.down2(y)
        
        return y1, y2, y


DINOv3_MODEL_FACTORIES = {
    "dinov3_s": dinov3_vits16,
    "dinov3_b": dinov3_vitb16,
    "dinov3_l": dinov3_vitl16,
}

DINOv3_ConvNext_FACTORIES = {
    "convnext_b": dinov3_convnext_base,
    "convnext_l": dinov3_convnext_large,
    "convnext_s": dinov3_convnext_small,
    "convnext_t": dinov3_convnext_tiny,
}

DINOv3_INTERACTION_INDEXES = {
    "dinov3_s": [2, 5, 8, 11],
    "dinov3_b": [2, 5, 8, 11],
    "dinov3_l": [4, 11, 17, 23],
    "dinov3_7b": [9, 19, 29, 39],
}

ConvNext_INTERACTION_INDEXES = {
    "convnext_t": [3, 3, 9, 3],
    "convnext_s": [3, 3, 27, 3],
    "convnext_b": [3, 3, 27, 3],
    "convnext_l": [3, 3, 27, 3],
}

DINOv3_MODEL_INFO = {
    "dinov3_s": {"embed_dim": 384, "depth": 12, "num_heads": 6, "params": "~22M"},
    "dinov3_b": {"embed_dim": 768, "depth": 12, "num_heads": 12, "params": "~86M"},
    "dinov3_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "params": "~300M"},
    "dinov3_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "params": "~7B"},
}

pretrained_urls = {
    "dinov3_s": r"/mnt/data/weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_sp": r"/mnt/data/weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_b": r"/mnt/data/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_l": r"/mnt/data/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_hp": r"/mnt/data/weights/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
}

convnext_pretrained_urls = {
    "convnext_t":  r"/mnt/data/weights/dinov3/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "convnext_s": r"/mnt/data/weights/dinov3/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "convnext_b":  r"/mnt/data/weights/dinov3/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "convnext_l": r"/mnt/data/weights/dinov3/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}

sat_pretrained_urls = {
    "dinov3_l": r"/mnt/data/weights/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
}