from typing import List
import torch
import torch.nn as nn

from ..anchor import Anchor

class YoloDetectionLayer(nn.Module):
    def __init__(self, img_size: int, stride: int = None,
            anchors: List = None):
        super().__init__()
        self.anchor = Anchor(anchors, img_size, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes detections
        Args:
            x (torch.Tensor): b x nA*(4+1) x gridy x gridx
        Returns:
            torch.Tensor: b x nA x gridy x gridx x (4+1)
        """
        bs, nCnA, gridy, gridx = x.shape
        nC = nCnA // self.anchor.num_anchors

        # bs x nA*(4+1) x gridy x gridx
        # => bs x nA x gridy x gridx x (4+1)

        # TODO check reshape order
        return x.reshape(bs, self.anchor.num_anchors, nC,
            gridy, gridx).permute(0, 1, 3, 4, 2)

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
