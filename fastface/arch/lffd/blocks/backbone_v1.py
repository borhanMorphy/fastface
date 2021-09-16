from typing import List

import torch
import torch.nn as nn

from .conv import conv3x3
from .resblock import ResBlock


class LFFDBackboneV1(nn.Module):
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
        self.res_block4 = ResBlock(64)

        # *small part
        self.conv3_dw = conv3x3(64, 64, stride=2, padding=0)
        self.relu3 = nn.ReLU()

        self.res_block5 = ResBlock(64)
        self.res_block6 = ResBlock(64)

        # *medium part
        self.conv4_dw = conv3x3(64, 128, stride=2, padding=0)
        self.relu4 = nn.ReLU()
        self.res_block7 = ResBlock(128)

        # *large part
        self.conv5_dw = conv3x3(128, 128, stride=2, padding=0)
        self.relu5 = nn.ReLU()
        self.res_block8 = ResBlock(128)
        self.res_block9 = ResBlock(128)
        self.res_block10 = ResBlock(128)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # *tiny part
        c1 = self.conv1_dw(x)  # 3 => 64
        r1 = self.relu1(c1)

        c2 = self.conv2_dw(r1)  # 64 => 64
        r2 = self.relu2(c2)

        r4, c4 = self.res_block1(r2, c2)  # 64 => 64
        r6, c6 = self.res_block2(r4, c4)  # 64 => 64
        r8, c8 = self.res_block3(r6, c6)  # 64 => 64
        r10, _ = self.res_block4(r8, c8)  # 64 => 64

        # *small part
        c11 = self.conv3_dw(r10)  # 64 => 64
        r11 = self.relu3(c11)

        r13, c13 = self.res_block5(r11, c11)  # 64 => 64
        r15, _ = self.res_block6(r13, c13)  # 64 => 64

        # *medium part
        c16 = self.conv4_dw(r15)  # 64 => 128
        r16 = self.relu4(c16)

        r18, _ = self.res_block7(r16, c16)  # 128 => 128

        # *large part
        c19 = self.conv5_dw(r18)  # 128 => 128
        r19 = self.relu5(c19)

        r21, c21 = self.res_block8(r19, c19)  # 128 => 128
        r23, c23 = self.res_block9(r21, c21)  # 128 => 128
        r25, _ = self.res_block10(r23, c23)  # 128 => 128

        return [r8, r10, r13, r15, r18, r21, r23, r25]
