from typing import List

import torch
import torch.nn as nn

from .conv import conv3x3
from .resblock import ResBlock


class LFFDBackboneV2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # *tiny part
        self.conv1_dw = conv3x3(in_channels, 64, stride=2, padding=0)
        self.relu1 = nn.ReLU()

        self.conv2_dw = conv3x3(64, 64, stride=2, padding=0)
        self.relu2 = nn.ReLU()

        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(64)
        self.res_block3 = ResBlock(64)

        # *small part
        self.conv3_dw = conv3x3(64, 64, stride=2, padding=0)
        self.relu3 = nn.ReLU()

        self.res_block4 = ResBlock(64)

        # *medium part
        self.conv4_dw = conv3x3(64, 64, stride=2, padding=0)
        self.relu4 = nn.ReLU()
        self.res_block5 = ResBlock(64)

        # *large part
        self.conv5_dw = conv3x3(64, 128, stride=2, padding=0)
        self.relu5 = nn.ReLU()
        self.res_block6 = ResBlock(128)

        # *large part
        self.conv6_dw = conv3x3(128, 128, stride=2, padding=0)
        self.relu6 = nn.ReLU()

        self.res_block7 = ResBlock(128)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # *tiny part
        c1 = self.conv1_dw(x)  # 3 => 64
        r1 = self.relu1(c1)

        c2 = self.conv2_dw(r1)  # 64 => 64
        r2 = self.relu2(c2)

        r4, c4 = self.res_block1(r2, c2)  # 64 => 64
        r6, c6 = self.res_block2(r4, c4)  # 64 => 64
        r8, _ = self.res_block3(r6, c6)  # 64 => 64

        # *small part
        c9 = self.conv3_dw(r8)  # 64 => 64
        r9 = self.relu3(c9)

        r11, _ = self.res_block4(r9, c9)  # 64 => 64

        # *medium part
        c12 = self.conv4_dw(r11)  # 64 => 64
        r12 = self.relu4(c12)

        r14, _ = self.res_block5(r12, c12)  # 64 => 64

        # *large part
        c15 = self.conv5_dw(r14)  # 64 => 128
        r15 = self.relu5(c15)
        r17, _ = self.res_block6(r15, c15)  # 128 => 128

        # *large part
        c18 = self.conv6_dw(r17)  # 128 => 128
        r18 = self.relu6(c18)
        r20, _ = self.res_block7(r18, c18)  # 128 => 128

        return [r8, r11, r14, r17, r20]
