from typing import List, Tuple
import torch
import torch.nn as nn
from .conv import conv3x3, conv1x1

class CSPBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv3x3_block_1 = conv3x3(features//2, features//2)
        self.conv3x3_block_2 = conv3x3(features//2, features//2)
        self.conv1x1_block = conv1x1(features, features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, num_of_features, _, _ = x.shape
        # split the input to 2 parts (channelwise)
        part = x[:, num_of_features//2:, :, :]

        residual = self.conv3x3_block_1(part)
        out = self.conv3x3_block_2(residual)
        out = torch.cat([residual, out], dim=1) # channelwise concatination
        residual = self.conv1x1_block(out)
        out = torch.cat([x, residual], dim=1) # channelwise concatination
        return out, residual