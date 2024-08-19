# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

from .arch_utils import IGConv, IGSample

from copy import deepcopy


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


@ARCH_REGISTRY.register()
class RDNIG(nn.Module):
    def __init__(
        self, G0=64, RDNkSize=3, RDNconfig='B',
        n_colors=3,
        # IGConv
        sk_kernel_size=3, implicit_dim=256,
        latent_layers=4, geo_ensemble=False,
    ):
        super(RDNIG, self).__init__()
        kSize = RDNkSize
        
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.out_dim = n_colors
        # Up-sampling net
        self.UPNet = IGConv(
            G0, sk_kernel_size, implicit_dim, latent_layers, geo_ensemble
        )

    def forward(self, x, scale):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1
        
        x = self.UPNet(x, scale)
        return x


@ARCH_REGISTRY.register()
class RDNIGPlus(nn.Module):
    def __init__(
        self, G0=64, RDNkSize=3, RDNconfig='B',
        n_colors=3,
        # IGConv
        sk_kernel_size=3, implicit_dim=256,
        latent_layers=4, geo_ensemble=False,
        max_s: int = 4,
        # IGSample
        return_skip=False,
    ):
        super(RDNIGPlus, self).__init__()
        kSize = RDNkSize
        
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.out_dim = n_colors
        # Up-sampling net
        self.UPNet = IGConv(
            G0, sk_kernel_size, implicit_dim, latent_layers, geo_ensemble, max_s
        )

        self.skip = IGSample(
            G0, sk_kernel_size, implicit_dim//2, latent_layers//2, geo_ensemble, max_s
        )
        self.return_skip = return_skip
        
    def forward(self, x, scale):
        inp = x
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1
        
        skip = self.skip(inp, x, scale)
        x = self.UPNet(x, scale) + skip
        if self.training and self.return_skip:
            return x, skip
        else:
            return x
