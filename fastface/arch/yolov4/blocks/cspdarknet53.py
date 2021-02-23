from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import (
    conv3x3_ds,
    conv3x3
)

from .csp import CSPBlock

class CSPDarknet53Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ds_1 = conv3x3_ds(3, 32)
        self.conv_ds_2 = conv3x3_ds(32, 64)
        self.csp_block_1 = CSPBlock(64)
        self.csp_block_2 = CSPBlock(128)
        self.csp_block_3 = CSPBlock(256)
        self.conv_3 = conv3x3(512, 512)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_ds_1(x)
        x = self.conv_ds_2(x)

        x, _ = self.csp_block_1(x)
        x = F.max_pool2d(x, 2, 2) # 2x2 stride:2 mp

        x, _ = self.csp_block_2(x)
        x = F.max_pool2d(x, 2, 2) # 2x2 stride:2 mp

        x, residual = self.csp_block_3(x)
        x = F.max_pool2d(x, 2, 2) # 2x2 stride:2 mp

        out = self.conv_3(x)

        return out, residual