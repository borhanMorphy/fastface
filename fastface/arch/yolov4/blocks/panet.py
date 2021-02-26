from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import conv3x3

class PANetTiny(nn.Module):
    def __init__(self, features: int):
        self.branch_1_conv_block_1 = conv3x3(features, features)
        self.branch_1_conv_block_2 = conv3x3(features, features//2)
        self.branch_1_conv_block_3 = conv3x3(features//2, features)

        self.branch_2_conv_block_1 = conv3x3(features//2, features//4)
        self.branch_2_conv_block_2 = conv3x3(features//2 + features//4, features//2)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # branch 1
        # x => CBL => CBL => residual1 => CBL => head1 (13x13xfeatures)
        x = self.branch_1_conv_block_1(x)
        residual_1 = self.branch_1_conv_block_2(x)
        out_1 = self.branch_1_conv_block_3(residual_1) # 13 x 13 x features

        # branch 2
        # residual1 => CBL => upsample => concat(residual) => CBL => head2 (26x26xfeatures//2)
        out = self.branch_2_conv_block_1(residual_1)
        out = F.interpolate(out, scale_factor=2) # 13x13 => 26x26
        out = torch.cat([out, residual], dim=1) # channelwise concatination

        out_2 = self.branch_2_conv_block_2(out) # 26 x 26 x features//2

        return out_1, out_2