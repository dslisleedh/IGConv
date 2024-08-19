"""
Coeds from BasicSR
https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/edsr_arch.py
"""
import torch
from torch import nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY

from einops import rearrange

from .arch_utils import IGConv, IGSample
from copy import deepcopy


@ARCH_REGISTRY.register()
class EDSRIG(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 # IGConv params
                 sk_kernel_size=3, 
                 implicit_dim=256,
                 latent_layers=4,
                 geo_ensemble=False,
        ):
        super(EDSRIG, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.upsample = Upsample(upscale, num_feat)
        # self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.upsample = IGConv(
            num_feat, sk_kernel_size, implicit_dim, latent_layers, geo_ensemble
        )

    def forward(self, x, scale):
        self.mean = self.mean.type_as(x)
        
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        
        x = self.upsample(res, scale)
        x = x / self.img_range + self.mean
        return x


@ARCH_REGISTRY.register()
class EDSRIGPlus(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 # IGConv params
                 sk_kernel_size=3, 
                 implicit_dim=256,
                 latent_layers=4,
                 geo_ensemble=True,
                 # IGSample params
                 return_skip=True,
        ):
        super(EDSRIGPlus, self).__init__()

        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale, pytorch_init=True)
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = IGConv(
            num_feat, sk_kernel_size, implicit_dim, latent_layers, geo_ensemble
        )
        self.skip = IGSample(
            dim=num_feat, kernel_size=3, implicit_dim=implicit_dim//2, latent_layers=latent_layers//2,
            geo_ensemble=geo_ensemble
        )
        self.return_skip = return_skip

    def forward(self, x, scale):
        inp = x
        self.mean = self.mean.type_as(x)
        
        x = (x - self.mean) * self.img_range
        x = self.conv_first(x)
        res = self.conv_after_body(self.body(x))
        res += x
        
        x = self.upsample(res, scale)
        x = x / self.img_range + self.mean
        
        skip = self.skip(inp, res, scale)
        x = x + skip
        if self.training and self.return_skip:
            return x, skip
        else:
            return x
