from typing import List, Tuple
import torch
import torch.nn as nn

from ..utils import AnchorGenerator

class YoloDetectionLayer(nn.Module):
    def __init__(self, img_size: int, stride: int = None,
            anchors: List = None):
        super().__init__()

        self.anchor_box_gen = AnchorGenerator(
            torch.tensor(anchors, dtype=torch.float32) * img_size,
            (int(img_size / stride), int(img_size / stride)), # gx, gy
            stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes detections

        Args:
            x (torch.Tensor): b x Na*(4+1) x gridy x gridx
        Returns:
            torch.Tensor: b x Na x gridy x gridx x (4+1)
        """
        batch_size = x.size(0)
        grid_y = x.size(-2)
        grid_x = x.size(-1)

        # b x anchors*(4+1) x grid_y x grid_x
        # => b x anchors x grid_y x grid_x x (4+1)
        out = x.reshape(
            batch_size, self.anchor_box_gen._anchors.size(0), -1,
            grid_y, grid_x).permute(0,1,3,4,2)

        return out

class YoloHead(nn.Module):
    def __init__(self, in_features: int, img_size: int,
            stride: int = None, anchors: List = None):
        super().__init__()

        self.conv = nn.Conv2d(in_features, int(len(anchors) * (4+1)),
            kernel_size=1, stride=1, padding=0)

        self.det_layer = YoloDetectionLayer(img_size,
            stride=stride, anchors=anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        logits = self.det_layer(x)
        return logits
