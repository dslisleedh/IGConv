import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DualStreamLoss(nn.Module):
    def __init__(self, alpha: float, loss_weight=1.0, reduction='mean'):
        super(DualStreamLoss, self).__init__()
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.criterion_lf = torch.nn.L1Loss(reduction=reduction)
        self.criterion_hf = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred_skip, pred_main, target):
        lf_loss = self.criterion_lf(pred_skip, target)  # GT - LF
        hf_loss = self.criterion_hf(pred_main, target)  # GT - LF - HF
        loss_total = self.alpha * lf_loss + hf_loss
        return self.loss_weight * loss_total
    