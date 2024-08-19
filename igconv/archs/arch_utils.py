import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import numpy as np

from basicsr.utils.registry import ARCH_REGISTRY
from functools import partial

import math


def make_coord(shape, ranges=None):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    ret = ret.flip(-1)
    return ret


class IGConv(nn.Module):
    def __init__(
            self, dim,
            # Own parameters
            kernel_size, implicit_dim: int = 256, latent_layers: int = 4,
            geo_ensemble: bool = True, max_s=4
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        assert implicit_dim % 2 == 0
        self.implicit_dim = implicit_dim
        self.latent_layers = latent_layers
        self.geo_ensemble = geo_ensemble
        self.max_s = max_s

        self.phase = nn.Conv2d(1, implicit_dim // 2, 1, 1)
        self.freq = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        self.amplitude = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        query_kernel_layers = []
        for _ in range(latent_layers):
            query_kernel_layers.append(
                nn.Conv2d(implicit_dim, implicit_dim, 1, 1, 0)
            )
            query_kernel_layers.append(nn.ReLU())

        query_kernel_layers.append(
            nn.Conv2d(implicit_dim, 3, 1, 1, 0)
        )
        self.query_kernel = nn.Sequential(*query_kernel_layers)
        self.resize = self._implicit_representation_latent

    def forward(self, x, scale, add_val=None):
        k_interp = self.resize(scale)
        if self.geo_ensemble:
            rgb = self._geo_ensemble(x, k_interp)
        else:
            rgb = F.conv2d(x, k_interp, bias=None, stride=1, padding=self.kernel_size // 2)
        if add_val is not None:
            rgb += add_val
        rgb = F.pixel_shuffle(rgb, scale)
        return rgb

    @staticmethod
    def _geo_ensemble(x, k_interp):
        k = k_interp
        k_hflip = k.flip([3])
        k_vflip = k.flip([2])
        k_hvflip = k.flip([2, 3])
        k_rot90 = torch.rot90(k, -1, [2, 3])
        k_rot90_hflip = k_rot90.flip([3])
        k_rot90_vflip = k_rot90.flip([2])
        k_rot90_hvflip = k_rot90.flip([2, 3])
        k = torch.cat([k, k_hflip, k_vflip, k_hvflip, k_rot90, k_rot90_hflip, k_rot90_vflip, k_rot90_hvflip], dim=0)
        ks = k.shape[-1]
        x = F.conv2d(x, k, bias=None, stride=1, padding=ks // 2)
        x = x.reshape(x.shape[0], 8, -1, x.shape[-2], x.shape[-1])
        x = x.mean(dim=1)
        return x

    def _implicit_representation_latent(self, scale):
        scale_phase = min(scale, self.max_s)
        r = torch.ones(1, 1, scale, scale).to(
            self.query_kernel[0].weight.device) / scale_phase * 2  # 2 / r following LIIF/LTE
        coords = make_coord((scale, scale)).unsqueeze(0).to(self.query_kernel[0].weight.device)
        freq = self.freq.repeat(1, 1, scale, scale)  # RGB RGB
        amplitude = self.amplitude.repeat(1, 1, scale, scale)
        coords = coords.permute(0, 3, 1, 2).contiguous()

        # Fourier basis
        coords = coords.repeat(freq.shape[0], 1, 1, 1)
        freq_1, freq_2 = freq.chunk(2, dim=1)
        freq = freq_1 * coords[:, :1] + freq_2 * coords[:, 1:]  # RGB
        phase = self.phase(r)  # To RGB
        freq = freq + phase  # RGB
        freq = torch.cat([torch.cos(np.pi * freq), torch.sin(np.pi * freq)],
                         dim=1)  # cos(R)cos(G)cos(B) sin(R)sin(G)sin(B)

        # 4. R(F_theta(.))
        k_interp = self.query_kernel(freq * amplitude)
        k_interp = rearrange(
            k_interp, '(Cin Kh Kw) RGB rh rw -> (RGB rh rw) Cin Kh Kw', Kh=self.kernel_size, Kw=self.kernel_size,
            Cin=self.dim
        )
        return k_interp

    def extra_repr(self):
        return f'dim={self.dim}, kernel_size={self.kernel_size}' + \
            f', \nimplicit_dim={self.implicit_dim}, latent_layers={self.latent_layers}, geo_ensemble={self.geo_ensemble}'

    def instantiate(self, scale):
        k = self._implicit_representation_latent(scale)
        device = k.device
        kernel_size = k.shape[-1]
        c_in = k.shape[1]
        c_out = k.shape[0]

        if self.geo_ensemble:
            c = k.shape[0]
            k_hflip = k.flip([3])
            k_vflip = k.flip([2])
            k_hvflip = k.flip([2, 3])
            k_rot90 = torch.rot90(k, -1, [2, 3])
            k_rot90_hflip = k_rot90.flip([3])
            k_rot90_vflip = k_rot90.flip([2])
            k_rot90_hvflip = k_rot90.flip([2, 3])
            k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8.

        self.__class__ = InstantiatedIGConv
        self.__init__(c_in, c_out, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.weight.data = k
        self.to(device)
        self.scale = scale


class InstantiatedIGConv(nn.Conv2d):
    def forward(self, x, scale=None, add_val=None):
        x = F.conv2d(x, self.weight, bias=None, stride=1, padding=self.padding)
        if add_val is not None:
            x += add_val
        x = F.pixel_shuffle(x, self.scale)
        return x


class IGConvDSSepRGB(nn.Module):
    def __init__(
            self, dim,
            # Own parameters
            kernel_size: int = 3,
            implicit_dim: int = 256,
            latent_layers: int = 4,
            geo_ensemble: bool = True,
            max_s=4,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.implicit_dim = implicit_dim
        self.latent_layers = latent_layers
        self.geo_ensemble = geo_ensemble
        self.max_s = max_s

        group = 2
        self.group = group
        implicit_dim = implicit_dim * 2
        self.phase = nn.Conv2d(1, implicit_dim // 2, 1, 1)
        self.freq = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        self.amplitude = nn.Parameter(
            torch.randn((dim * kernel_size ** 2), implicit_dim, 1, 1) * 0.02, requires_grad=True)
        query_kernel_layers = []
        for _ in range(latent_layers):
            query_kernel_layers.append(
                nn.Conv2d(implicit_dim, implicit_dim, 1, 1, 0, groups=group)
            )
            query_kernel_layers.append(nn.ReLU())
        query_kernel_layers.append(
            nn.Conv2d(implicit_dim, 12, 1, 1, 0, groups=group, bias=False)
        )
        self.query_kernel = nn.Sequential(*query_kernel_layers)
        for p in self.query_kernel.parameters():
            torch.nn.init.trunc_normal_(p, std=0.01)
        # to initialize RGB delta same
        with torch.no_grad():
            kernel_offset = torch.nn.init.trunc_normal_(torch.zeros(2, implicit_dim // 2, 1, 1), std=0.01)
            kernel_offset_rgb = kernel_offset.repeat(3, 1, 1, 1)
            kernel_scope = torch.nn.init.zeros_(torch.zeros(2, implicit_dim // 2, 1, 1))
            kernel_scope_rgb = kernel_scope.repeat(3, 1, 1, 1)
            kernel = torch.cat([kernel_offset_rgb, kernel_scope_rgb], dim=0)
            self.query_kernel[-1].weight.data = kernel

        self.resize = self._implicit_representation_latent

    def forward(self, x, scale):
        k_interp = self.resize(scale)
        if self.geo_ensemble:
            xyxy = self._geo_ensemble(x, k_interp)
        else:
            xyxy = F.conv2d(x, k_interp, bias=None, stride=1, padding=self.kernel_size // 2)
        xy_offset, xy_scope = xyxy.chunk(2, dim=1)
        return xy_offset, xy_scope

    @staticmethod
    def _geo_ensemble(x, k_interp):
        k = k_interp
        k_hflip = k.flip([3])
        k_vflip = k.flip([2])
        k_hvflip = k.flip([2, 3])
        k_rot90 = torch.rot90(k, -1, [2, 3])
        k_rot90_hflip = k_rot90.flip([3])
        k_rot90_vflip = k_rot90.flip([2])
        k_rot90_hvflip = k_rot90.flip([2, 3])
        k = (k + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8
        ks = k.shape[-1]
        x = F.conv2d(x, k, bias=None, stride=1, padding=ks // 2)
        return x

    def _implicit_representation_latent(self, scale):
        scale_phase = min(scale, self.max_s)
        r = torch.ones(1, 1, scale, scale).to(
            self.query_kernel[0].weight.device) / scale_phase * 2  # 2 / r following LIIF/LTE
        coords = make_coord((scale, scale)).unsqueeze(0).to(self.query_kernel[0].weight.device)
        freq = self.freq.repeat(1, 1, scale, scale)  # OffsetScopeOffsetScope
        amplitude = self.amplitude.repeat(1, 1, scale, scale)
        coords = coords.permute(0, 3, 1, 2).contiguous()

        coords = coords.repeat(freq.shape[0], 1, 1, 1)
        freq_1, freq_2 = freq.chunk(2, dim=1)
        freq = freq_1 * coords[:, :1] + freq_2 * coords[:, 1:]  # OffsetScope
        phase = self.phase(r)
        freq = freq + phase
        freq = torch.cat([torch.cos(np.pi * freq), torch.sin(np.pi * freq)],
                         dim=1)  # cos(Offset)cos(Scope) sin(Offset)sin(Scope)
        freq = rearrange(freq, 'b (g d) h w -> b (d g) h w', d=2)  # cos(Offset)sin(Offset) cos(Scope)sin(Scope)

        k_interp = self.query_kernel(freq * amplitude)
        k_interp = rearrange(
            k_interp, '(Cin Kh Kw) RGB rh rw -> (RGB rh rw) Cin Kh Kw', Kh=self.kernel_size, Kw=self.kernel_size,
            Cin=self.dim
        )
        return k_interp


class IGSample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample':
    https://arxiv.org/abs/2308.15085
    https://github.com/tiny-smart/dysample

    We implemented IGSample based on NeoSR's DySample implementation:
    https://github.com/muslll/neosr/blob/master/neosr/archs/arch_util.py
    """

    def __init__(self, dim, kernel_size=3, implicit_dim=128, latent_layers=2, geo_ensemble=True, max_s=4):
        super().__init__()

        self.convs = IGConvDSSepRGB(
            dim, kernel_size, implicit_dim=implicit_dim, latent_layers=latent_layers,
            geo_ensemble=geo_ensemble, max_s=max_s
        )

    def pos(self, scale):
        h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1) / scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, 3, 1)
            .reshape(1, -1, 1, 1)
        )

    def forward(self, x: torch.Tensor, feat: torch.Tensor, scale: int):
        offset, scope = self.convs(feat, scale)
        pos = self.pos(scale).to(x.device)

        offset = offset * scope.sigmoid() * 0.5 + pos

        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5

        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
            .to(x.device, non_blocking=True)
        )
        normalizer = torch.tensor(
            [W, H], dtype=x.dtype, device=x.device, pin_memory=True
        ).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = (
            F.pixel_shuffle(coords.reshape(B, -1, H, W), scale)
            .view(B, 2, -1, scale * H, scale * W)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        output = F.grid_sample(
            x.reshape(B * 3, -1, H, W),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(B, -1, scale * H, scale * W)

        return output

    def extra_repr(self) -> str:
        return f'dim={self.convs.dim}, kernel_size={self.convs.kernel_size}' + \
            f', \nimplicit_dim={self.convs.implicit_dim}, latent_layers={self.convs.latent_layers}, geo_ensemble={self.convs.geo_ensemble}'


def test_direct_metrics(model, input_shape, scale, n_repeat=100, use_float16=True):
    from torch.backends import cudnn
    import tqdm
    import numpy as np
    from contextlib import nullcontext

    cudnn.benchmark = True

    print(f'CUDNN Benchmark: {cudnn.benchmark}')
    if use_float16:
        context = torch.cuda.amp.autocast
        print('Using AMP(FP16) for testing ...')
    else:
        context = nullcontext
        print('Using FP32 for testing ...')

    x = torch.FloatTensor(*input_shape).uniform_(0., 1.)
    x = x.cuda()
    print(f'Input shape: {x.shape}')
    model = model.cuda()
    model.eval()

    with context():
        with torch.inference_mode():
            print('warmup ...')
            for _ in tqdm.tqdm(range(100)):
                model(x, scale)  # Make sure CUDNN to find proper algorithms, especially for convolutions.
                torch.cuda.synchronize()

            print('testing ...')
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = np.zeros((n_repeat, 1))

            for rep in tqdm.tqdm(range(n_repeat)):
                starter.record()
                model(x, scale)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

    avg = np.sum(timings) / n_repeat
    med = np.median(timings)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('------------ Results ------------')
    print(f'Average time: {avg:.5f} ms')
    print(f'Median time: {med:.5f} ms')
    print(f'Maximum GPU memory Occupancy: {torch.cuda.max_memory_allocated() / 1024 ** 2:.5f} MB')
    print(f'Params: {params / 1000}K')  # For convenience and sanity check.
    print('---------------------------------')
