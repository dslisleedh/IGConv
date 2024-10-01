import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def case_1(x, k):
    x = torch.from_numpy(x).float().requires_grad_(False)
    k = torch.from_numpy(k).float().requires_grad_(True)

    f_identity = F.conv2d(x, k, padding=1)
    f_hflip = F.conv2d(torch.flip(x, [3]), k, padding=1).flip([3])
    f_vflip = F.conv2d(torch.flip(x, [2]), k, padding=1).flip([2])
    f_hvflip = F.conv2d(torch.flip(torch.flip(x, [3]), [2]), k, padding=1).flip([3, 2])
    f_rot90 = torch.rot90(
        F.conv2d(torch.rot90(x, -1, [2, 3]), k, bias=None, stride=1, padding=1),
        1, [2, 3]
    )
    f_rot90_hflip = torch.rot90(
        F.conv2d(torch.rot90(x, -1, [2, 3]).flip([3]), k, bias=None, stride=1, padding=1).flip([3]),
        1, [2, 3]
    )
    f_rot90_vflip = torch.rot90(
        F.conv2d(torch.rot90(x, -1, [2, 3]).flip([2]), k, bias=None, stride=1, padding=1).flip([2]),
        1, [2, 3]
    )
    f_rot90_hv = torch.rot90(
        F.conv2d(torch.rot90(x, -1, [2, 3]).flip([2, 3]), k, bias=None, stride=1, padding=1).flip([2, 3]),
        1, [2, 3]
    )

    f_bar = f_identity + f_hflip + f_vflip + f_hvflip + f_rot90 + f_rot90_hflip + f_rot90_vflip + f_rot90_hv
    f_bar /= 8.

    loss = f_bar.sum()
    loss.backward()
    grad = k.grad.clone()
    return f_bar, grad


def case_2(x, k):
    x = torch.from_numpy(x).float().requires_grad_(False)
    k = torch.from_numpy(k).float().requires_grad_(True)

    k_identity = k.clone()
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, -1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])

    f_identity = F.conv2d(x, k_identity, padding=1)
    f_hflip = F.conv2d(x, k_hflip, padding=1)
    f_vflip = F.conv2d(x, k_vflip, padding=1)
    f_hvflip = F.conv2d(x, k_hvflip, padding=1)
    f_rot90 = F.conv2d(x, k_rot90, padding=1)
    f_rot90_hflip = F.conv2d(x, k_rot90_hflip, padding=1)
    f_rot90_vflip = F.conv2d(x, k_rot90_vflip, padding=1)
    f_rot90_hvflip = F.conv2d(x, k_rot90_hvflip, padding=1)

    f_bar = f_identity + f_hflip + f_vflip + f_hvflip + f_rot90 + f_rot90_hflip + f_rot90_vflip + f_rot90_hvflip
    f_bar /= 8.

    loss = f_bar.sum()
    loss.backward()
    grad = k.grad.clone()
    return f_bar, grad


def case_3(x, k):
    x = torch.from_numpy(x).float().requires_grad_(False)
    k = torch.from_numpy(k).float().requires_grad_(True)

    k_identity = k.clone()
    k_hflip = k.flip([3])
    k_vflip = k.flip([2])
    k_hvflip = k.flip([2, 3])
    k_rot90 = torch.rot90(k, 1, [2, 3])
    k_rot90_hflip = k_rot90.flip([3])
    k_rot90_vflip = k_rot90.flip([2])
    k_rot90_hvflip = k_rot90.flip([2, 3])

    k_bar = (k_identity + k_hflip + k_vflip + k_hvflip + k_rot90 + k_rot90_hflip + k_rot90_vflip + k_rot90_hvflip) / 8.
    f_bar = F.conv2d(x, k_bar, padding=1)

    loss = f_bar.sum()
    loss.backward()
    grad = k.grad.clone()
    return f_bar, grad


if __name__ == '__main__':
    x_np = np.random.randn(1, 32, 64, 64)
    k_np = np.random.randn(32, 32, 3, 3)

    f1, g1 = case_1(x_np, k_np)
    f2, g2 = case_2(x_np, k_np)
    f3, g3 = case_3(x_np, k_np)

    assert not torch.allclose(g1, torch.zeros_like(g1))

    # Check if the results are the same.
    print(torch.allclose(f1, f2, atol=1e-4))
    print(torch.allclose(f1, f3, atol=1e-4))
    print(torch.allclose(g1, g2, atol=1e-4))
    print(torch.allclose(g1, g3, atol=1e-4))

