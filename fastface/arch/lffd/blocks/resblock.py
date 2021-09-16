from typing import Tuple

import torch
import torch.nn as nn

from .conv import conv3x3


class ResBlock(nn.Module):
    def __init__(self, features: int):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(features, features)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(features, features)
        self.relu2 = nn.ReLU()

    def forward(
        self, activated_input: torch.Tensor, residual_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(activated_input)  # c(i) => c(i+1)
        x = self.relu1(x)
        x = self.conv2(x)  # c(i+1) => c(i+2)
        residual_output = x + residual_input  # residual
        activated_output = self.relu2(residual_output)
        return activated_output, residual_output
