from typing import Tuple

import torch
import torch.nn as nn

from .anchor import Anchor
from .conv import conv1x1


class LFFDHead(nn.Module):
    def __init__(
        self,
        head_idx: int,
        infeatures: int,
        features: int,
        rf_size: int,
        rf_start_offset: int,
        rf_stride: int,
        num_classes: int = 1,
    ):
        super().__init__()
        self.head_idx = head_idx
        self.num_classes = num_classes
        self.anchor = Anchor(rf_stride, rf_start_offset, rf_size)

        self.det_conv = nn.Sequential(conv1x1(infeatures, features), nn.ReLU())

        self.cls_head = nn.Sequential(
            conv1x1(features, features), nn.ReLU(), conv1x1(features, self.num_classes)
        )

        self.reg_head = nn.Sequential(
            conv1x1(features, features), nn.ReLU(), conv1x1(features, 4)
        )

        def conv_xavier_init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(conv_xavier_init)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.det_conv(features)

        cls_logits = self.cls_head(data)
        # (b,1,h,w)
        reg_logits = self.reg_head(data)
        # (b,c,h,w)
        return reg_logits, cls_logits

    def logits_to_boxes(self, reg_logits: torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,fh,fw,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        _, fh, fw, _ = reg_logits.shape

        rf_centers = self.anchor.rf_centers[:fh, :fw]
        # rf_centers: fh,fw,4 cx1,cy1,cx1,cy1

        # reg_logits[:, :, :, 0] = torch.clamp(reg_logits[:, :, :, 0], 0, fw*self.rf_stride)
        # reg_logits[:, :, :, 1] = torch.clamp(reg_logits[:, :, :, 1], 0, fh*self.rf_stride)
        # reg_logits[:, :, :, 2] = torch.clamp(reg_logits[:, :, :, 2], 0, fw*self.rf_stride)
        # reg_logits[:, :, :, 3] = torch.clamp(reg_logits[:, :, :, 3], 0, fh*self.rf_stride)

        return rf_centers - reg_logits * self.anchor.rf_normalizer
