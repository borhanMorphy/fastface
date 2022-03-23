import torch
import torch.nn as nn


class CenterFaceHead(nn.Module):
    def __init__(self, in_features: int, num_landmarks: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
        )
        self.cls_head = nn.Conv2d(in_features, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(in_features, 2, kernel_size=1)
        self.wh_head = nn.Conv2d(in_features, 2, kernel_size=1)
        self.landmark_head = nn.Conv2d(in_features, num_landmarks * 2, kernel_size=1)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        logits = self.conv_block(fmap)
        cls_logits = self.cls_head(logits)
        offset_regs = self.offset_head(logits)
        wh_regs = self.wh_head(logits)
        l_regs = self.landmark_head(logits)

        return (
            torch.cat([cls_logits, wh_regs, offset_regs, l_regs], dim=1)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
