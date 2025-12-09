import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
import argparse
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from torchvision.models.resnet import resnet34, ResNet34_Weights, resnet50
from torchvision.models.vision_transformer import vit_b_16, VisionTransformer
from torchvision.models.swin_transformer import swin_v2_b
from torchvision import transforms

from .vmamba.vmamba import VSSM, LayerNorm2d

from .configs.config import get_config

mamba_version_weights = {"vssm_base_224.yaml":"/mnt/data/weights/vssm_base_0229_ckpt_epoch_237.pth",
                         "vssm_small_224.yaml":"/mnt/data/weights/vssm_small_0229_ckpt_epoch_222.pth",
                         "vssm_tiny_224.yaml":"/mnt/data/weights/vssm_tiny_0230_ckpt_epoch_262.pth"}

parser = argparse.ArgumentParser(description="VSSMamba")
parser.add_argument('--cfg', type=str, default='./configs/vssm/vssm_tiny_224.yaml')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+')

mparas = parser.parse_args()
config = get_config(mparas)


class Backbone_VSSM(VSSM):
    def __init__(self, 
                 out_indices=(0, 1, 2, 3), 
                 pretrained=mamba_version_weights["vssm_tiny_224.yaml"],norm_layer='ln'):
        # norm_layer='ln'
        # kwargs.update(norm_layer=norm_layer)
        super().__init__( patch_size=config.MODEL.VSSM.PATCH_SIZE, 
                        in_chans=config.MODEL.VSSM.IN_CHANS, 
                        num_classes=config.MODEL.NUM_CLASSES, 
                        depths=config.MODEL.VSSM.DEPTHS, 
                        dims=config.MODEL.VSSM.EMBED_DIM, 
                        # ===================
                        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
                        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
                        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
                        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
                        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
                        ssm_conv=config.MODEL.VSSM.SSM_CONV,
                        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
                        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
                        ssm_init=config.MODEL.VSSM.SSM_INIT,
                        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
                        # ===================
                        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
                        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
                        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
                        # ===================
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        patch_norm=config.MODEL.VSSM.PATCH_NORM,
                        norm_layer=config.MODEL.VSSM.NORM_LAYER,
                        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
                        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
                        gmlp=config.MODEL.VSSM.GMLP,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y
        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs

# parser = argparse.ArgumentParser(description="VSSMamba_tiny")
# parser.add_argument('--cfg', type=str, default='/home/jq/Code/VMamba/lccdmamba/configs/vssm/vssm_tiny_224.yaml')
# parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+')

# mparas = parser.parse_args()
# config = get_config(mparas)

# class Backbone_VSSM_tiny(VSSM):
#     def __init__(self, 
#                  out_indices=(0, 1, 2, 3), 
#                  pretrained=mamba_version_weights["vssm_tiny_224.yaml"],norm_layer='ln'):
#         # norm_layer='ln'
#         # kwargs.update(norm_layer=norm_layer)
#         super().__init__( patch_size=config.MODEL.VSSM.PATCH_SIZE, 
#                         in_chans=config.MODEL.VSSM.IN_CHANS, 
#                         num_classes=config.MODEL.NUM_CLASSES, 
#                         depths=config.MODEL.VSSM.DEPTHS, 
#                         dims=config.MODEL.VSSM.EMBED_DIM, 
#                         # ===================
#                         ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#                         ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#                         ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#                         ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#                         ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#                         ssm_conv=config.MODEL.VSSM.SSM_CONV,
#                         ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#                         ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#                         ssm_init=config.MODEL.VSSM.SSM_INIT,
#                         forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#                         # ===================
#                         mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#                         mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#                         mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#                         # ===================
#                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                         patch_norm=config.MODEL.VSSM.PATCH_NORM,
#                         norm_layer=config.MODEL.VSSM.NORM_LAYER,
#                         downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#                         patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#                         gmlp=config.MODEL.VSSM.GMLP,
#                         use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        
#         self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
#         _NORMLAYERS = dict(
#             ln=nn.LayerNorm,
#             ln2d=LayerNorm2d,
#             bn=nn.BatchNorm2d,
#         )
#         norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
#         self.out_indices = out_indices
#         for i in out_indices:
#             layer = norm_layer(self.dims[i])
#             layer_name = f'outnorm{i}'
#             self.add_module(layer_name, layer)

#         del self.classifier
#         self.load_pretrained(pretrained)

#     def load_pretrained(self, ckpt=None, key="model"):
#         if ckpt is None:
#             return
        
#         try:
#             _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
#             print(f"Successfully load ckpt {ckpt}")
#             incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
#             print(incompatibleKeys)        
#         except Exception as e:
#             print(f"Failed loading checkpoint form {ckpt}: {e}")

#     def forward(self, x):
#         def layer_forward(l, x):
            
#             x = l.blocks(x)
#             y = l.downsample(x)
#             return x, y
#         x = self.patch_embed(x)
#         outs = []
#         for i, layer in enumerate(self.layers):
#             o, x = layer_forward(layer, x) # (B, H, W, C)
#             if i in self.out_indices:
#                 norm_layer = getattr(self, f'outnorm{i}')
#                 out = norm_layer(o)
#                 if not self.channel_first:
#                     out = out.permute(0, 3, 1, 2).contiguous()
#                 outs.append(out)

#         if len(self.out_indices) == 0:
#             return x
        
#         return outs


# parser = argparse.ArgumentParser(description="VSSMamba")
# parser.add_argument('--cfg', type=str, default='/home/jq/Code/VMamba/lccdmamba/configs/vssm/vssm_small_224.yaml')
# parser.add_argument(
#         "--opts",
#         help="Modify config options by adding 'KEY VALUE' pairs. ",
#         default=None,
#         nargs='+')

# mparas = parser.parse_args()
# config = get_config(mparas)

# class Backbone_VSSM_samll(VSSM):
#     def __init__(self, 
#                  out_indices=(0, 1, 2, 3), 
#                  pretrained=mamba_version_weights["vssm_small_224.yaml"],norm_layer='ln'):
#         # norm_layer='ln'
#         # kwargs.update(norm_layer=norm_layer)
#         super().__init__( patch_size=config.MODEL.VSSM.PATCH_SIZE, 
#                         in_chans=config.MODEL.VSSM.IN_CHANS, 
#                         num_classes=config.MODEL.NUM_CLASSES, 
#                         depths=config.MODEL.VSSM.DEPTHS, 
#                         dims=config.MODEL.VSSM.EMBED_DIM, 
#                         # ===================
#                         ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
#                         ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
#                         ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
#                         ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
#                         ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
#                         ssm_conv=config.MODEL.VSSM.SSM_CONV,
#                         ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
#                         ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
#                         ssm_init=config.MODEL.VSSM.SSM_INIT,
#                         forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
#                         # ===================
#                         mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
#                         mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
#                         mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
#                         # ===================
#                         drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                         patch_norm=config.MODEL.VSSM.PATCH_NORM,
#                         norm_layer=config.MODEL.VSSM.NORM_LAYER,
#                         downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
#                         patchembed_version=config.MODEL.VSSM.PATCHEMBED,
#                         gmlp=config.MODEL.VSSM.GMLP,
#                         use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        
#         self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
#         _NORMLAYERS = dict(
#             ln=nn.LayerNorm,
#             ln2d=LayerNorm2d,
#             bn=nn.BatchNorm2d,
#         )
#         norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)        
        
#         self.out_indices = out_indices
#         for i in out_indices:
#             layer = norm_layer(self.dims[i])
#             layer_name = f'outnorm{i}'
#             self.add_module(layer_name, layer)

#         del self.classifier
#         self.load_pretrained(pretrained)

#     def load_pretrained(self, ckpt=None, key="model"):
#         if ckpt is None:
#             return
        
#         try:
#             _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
#             print(f"Successfully load ckpt {ckpt}")
#             incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
#             print(incompatibleKeys)        
#         except Exception as e:
#             print(f"Failed loading checkpoint form {ckpt}: {e}")

#     def forward(self, x):
#         def layer_forward(l, x):
            
#             x = l.blocks(x)
#             y = l.downsample(x)
#             return x, y
#         x = self.patch_embed(x)
#         outs = []
#         for i, layer in enumerate(self.layers):
#             o, x = layer_forward(layer, x) # (B, H, W, C)
#             if i in self.out_indices:
#                 norm_layer = getattr(self, f'outnorm{i}')
#                 out = norm_layer(o)
#                 if not self.channel_first:
#                     out = out.permute(0, 3, 1, 2).contiguous()
#                 outs.append(out)

#         if len(self.out_indices) == 0:
#             return x
        
#         return outs