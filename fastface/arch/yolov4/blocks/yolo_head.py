from typing import List
import torch
import torch.nn as nn

from .anchor import Anchor

class YoloHead(nn.Module):
    def __init__(self, in_features: int, stride: int, anchors: List, img_size: int):
        super().__init__()

        self.img_size = img_size

        self.conv = nn.Conv2d(in_features, int(len(anchors) * (4+1)),
            kernel_size=1, stride=1, padding=0)

        self.anchor = Anchor(anchors, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes logits
        Args:
            x (torch.Tensor): b x C x fh x fw
        Returns:
            torch.Tensor: b x nA*(4+1) x fh x fw
        """
        return self.conv(x)

    def logits_to_boxes(self, reg_logits: torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): b x fh x fw x nA x 4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,nA,4 as cx,cy,w,h
        """
        _, fh, fw, _, _ = reg_logits.shape
        priors = self.anchor(fh, fw).permute(1,2,0,3)
        # priors: (fh x fw x nA x 4) as cx, cy, w, h

        # cell offsets between 0 -> grid_w | 0 -> grid_h
        cell_x = priors[..., 0] / self.anchor.stride - 0.5
        cell_y = priors[..., 1] / self.anchor.stride - 0.5

        # anchor sizes between almost 0 -> 1
        pw = priors[..., 2] / self.img_size
        ph = priors[..., 3] / self.img_size

        bx = torch.sigmoid(reg_logits[..., 0]) + cell_x
        by = torch.sigmoid(reg_logits[..., 1]) + cell_y
        bw = torch.exp(reg_logits[..., 2]) * pw
        bh = torch.exp(reg_logits[..., 3]) * ph

        pred_boxes = torch.stack([bx, by, bw, bh], dim=4).contiguous()
        pred_boxes[..., :2] *= self.anchor.stride # rescale centers
        pred_boxes[..., 2:] *= self.img_size # rescale width and height. 

        return pred_boxes