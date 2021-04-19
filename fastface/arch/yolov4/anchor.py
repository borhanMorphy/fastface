from typing import List
import torch
import torch.nn as nn

from ...utils.box import (
    generate_grids,
    xyxy2cxcywh,
    cxcywh2xyxy
)

class Anchor(nn.Module):
    def __init__(self, anchors: List, img_size: int, stride: int):
        super().__init__()
        # anchors: between 0 < _ < 1
        # pylint: disable=not-callable
        self.anchor_sizes = (torch.tensor(anchors) * img_size) / stride # between 0 < _ < max_grid
        self.stride = stride
        self.img_size = img_size
        self.num_anchors = len(anchors)

    @torch.jit.unused
    def estimated_forward(self, imgh: int, imgw: int) -> torch.Tensor:
        """Estimates anchors using image dimensions

        Args:
            imgh (int): image height
            imgw (int): image width

        Returns:
            torch.Tensor: anchors with shape (nA x fh x fw x 4) as xmin, ymin, xmax, ymax
        """
        fh = imgh // self.stride
        fw = imgw // self.stride
        return self.forward(fh, fw)

    def forward(self, fh: int, fw: int) -> torch.Tensor:
        """takes feature map h and w and reconstructs prior boxes as tensor
        Args:
            fh (int): hight of the feature map
            fw (int): width of the feature map
        Returns:
            torch.Tensor: prior boxes as (nA x fh x fw x 4) (xmin, ymin, xmax, ymax)
        """
        grids = generate_grids(fh, fw).unsqueeze(0).repeat(self.num_anchors, 1, 1, 1)

        wh = torch.repeat_interleave(self.anchor_sizes, fh*fw, dim=0).reshape(self.num_anchors, fh, fw, 2)
        prior_boxes = torch.cat([grids, wh], dim=3)
        prior_boxes[:, :, :, :2] += .5 # adjust to center

        prior_boxes *= self.stride

        return cxcywh2xyxy(
            prior_boxes.reshape(self.num_anchors*fh*fw, 4)).reshape(
                self.num_anchors, fh, fw, 4)

    def logits_to_boxes(self, reg_logits: torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,nA,grid_y,grid_x,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        batch_size = reg_logits.size(0)
        num_anchors = reg_logits.size(1)
        grid_y, grid_x = reg_logits.shape[2:4]
        device = reg_logits.device
        dtype = reg_logits.dtype

        prior_boxes = self.forward(grid_y, grid_x).reshape(num_anchors*grid_y*grid_x, 4).to(device, dtype)
        prior_boxes = xyxy2cxcywh(prior_boxes).reshape(num_anchors, grid_y, grid_x, 4) / self.stride

        prior_boxes[:, :, :, :2] -= 0.5

        # nA,gy,gx,4 => bs,nA,gy,gx,4
        prior_boxes = prior_boxes.repeat(batch_size, 1, 1, 1, 1)

        # bx = (sigmoid(tx) + x)
        # by = (sigmoid(ty) + y)
        # bw = exp(tw) * w
        # bh = exp(th) * h

        pred_xy = torch.sigmoid(reg_logits[:, :, :, :, :2]) + prior_boxes[:, :, :, :, :2]
        pred_w = torch.exp(reg_logits[:, :, :, :, 2]) * self.anchor_sizes[:, 0].reshape(1, self.num_anchors, 1, 1)
        pred_h = torch.exp(reg_logits[:, :, :, :, 3]) * self.anchor_sizes[:, 1].reshape(1, self.num_anchors, 1, 1)

        pred_boxes = torch.cat([pred_xy, pred_w.unsqueeze(4), pred_h.unsqueeze(4)], dim=4) * self.stride

        return cxcywh2xyxy(
            pred_boxes.reshape(batch_size*num_anchors*grid_y*grid_x, 4)).reshape(
                batch_size, num_anchors, grid_y, grid_x, 4)
