import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List
from .conv import conv1x1
from ..anchor import Anchor

class DetectionHead(nn.Module):
    def __init__(self, head_idx:int, infeatures:int, features:int,
            rf_size:int, rf_start_offset:int, rf_stride:int, num_classes:int=1):
        super(DetectionHead,self).__init__()
        self.head_idx = head_idx
        self.num_classes = num_classes
        self.anchor_box_gen = Anchor(rf_stride, rf_start_offset, rf_size)

        self.det_conv = nn.Sequential(
            conv1x1(infeatures, features), nn.ReLU())

        self.cls_head = nn.Sequential(
            conv1x1(features, features),
            nn.ReLU(),
            conv1x1(features, self.num_classes))

        self.reg_head = nn.Sequential(
            conv1x1(features, features),
            nn.ReLU(),
            conv1x1(features, 4))

        def conv_xavier_init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(conv_xavier_init)

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.det_conv(x)

        cls_logits = self.cls_head(data)
        # (b,c,h,w)
        reg_logits = self.reg_head(data)
        # (b,c,h,w)
        return cls_logits,reg_logits