from typing import List
import torch
import torch.nn as nn
from .conv import conv3x3, conv1x1

class CSPBlock(nn.Module):
    def __init__(self, features: List[int]):
        super().__init__()
        assert len(features) == 4, "CSP Block must contain 4 features"
        self.conv3x3_block_1 = conv3x3(features[0], features[1])
        self.conv3x3_block_2 = conv3x3(features[1], features[2])
        self.conv1x1_block = conv1x1(features[2], features[3])

    def forward(self, x: torch.Tensor):
        _, num_of_features, _, _ = x.shape
        # split the input to 2 parts (channelwise)
        part_1 = x[:, : num_of_features//2, :, :]
        part_2 = x[:, num_of_features//2:, :, :]

        residual = self.conv3x3_block_1(part_2)
        out = self.conv3x3_block_2(residual)
        out = torch.cat([residual, out], dim=1) # channelwise concatination
        out = self.conv1x1_block(out)
        out = torch.cat([part_1, out], dim=1) # channelwise concatination
        return out
