from typing import Tuple

import torch
import torch.nn as nn


class CenterFaceHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
        )
        self.cls_head = nn.Conv2d(in_features, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(in_features, 2, kernel_size=1)
        self.wh_head = nn.Conv2d(in_features, 2, kernel_size=1)

    def forward(
        self, fmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.conv_block(fmap)
        cls_logits = self.cls_head(logits).squeeze(1).contiguous()
        offset_regs = self.offset_head(logits).permute(0, 2, 3, 1).contiguous()
        wh_regs = self.wh_head(logits).permute(0, 2, 3, 1).contiguous()

        return cls_logits, offset_regs, wh_regs
