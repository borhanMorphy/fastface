from typing import Tuple
import torch

from ...utils.box import (
    xyxy2cxcywh,
    cxcywh2xyxy
)

class Anchor():
    def __init__(self, anchors: torch.Tensor, grids: Tuple[int, int], stride: int):
        self._anchors = anchors / stride
        self._cached_grids = list(grids) # gx, gy
        self._device = 'cpu'
        self._dtype = torch.float32
        self._stride = stride

        num_anchors = self._anchors.size(0)

        grid_x = self._cached_grids[0]
        grid_y = self._cached_grids[1]

        grids = self.generate_grids(grid_y, grid_x,
            device=self._device, dtype=self._dtype).unsqueeze(0).repeat(num_anchors,1,1,1)

        wh = torch.repeat_interleave(self._anchors, grid_y*grid_x, dim=0).reshape(num_anchors, grid_y, grid_x, 2)
        prior_boxes = torch.cat([grids, wh], dim=-1)
        prior_boxes[:, :, :, :2] += .5 # adjust to center
        prior_boxes *= self._stride

        self._prior_boxes = cxcywh2xyxy(prior_boxes.reshape(-1,4)).reshape(num_anchors, grid_y, grid_x, 4)


    def __call__(self, grid_y:int, grid_x:int, device:str='cpu',
            dtype:torch.dtype=torch.float32) -> torch.Tensor:
        """takes feature map h and w and reconstructs rf anchors as tensor
        Args:
            grid_y (int): grid hight
            grid_x (int): grid width
            device (str, optional): selected device to anchors will be generated. Defaults to 'cpu'.
            dtype (torch.dtype, optional): selected dtype to anchors will be generated. Defaults to torch.float32.
  
        Returns:
            torch.Tensor: rf anchors as (nA x grid_y x grid_x x 4) (xmin, ymin, xmax, ymax)
        """

        if self._device != device or self._dtype != dtype:
            self._anchors = self._anchors.to(device, dtype)
            self._prior_boxes = self._prior_boxes.to(device, dtype)
            self._device = device
            self._dtype = dtype

        if self._cached_grids[0] == grid_x and self._cached_grids[1] == grid_y:
            return self._prior_boxes.clone()

        self._cached_grids[0] = grid_x
        self._cached_grids[1] = grid_y

        num_anchors = self._anchors.size(0)

        grids = self.generate_grids(grid_y, grid_x,
            device=device, dtype=dtype).unsqueeze(0).repeat(num_anchors,1,1,1)

        wh = torch.repeat_interleave(self._anchors, grid_y*grid_x, dim=0).reshape(num_anchors, grid_y, grid_x, 2)
        prior_boxes = torch.cat([grids, wh], dim=-1)
        prior_boxes[:, :, :, :2] += .5 # adjust to center
        prior_boxes *= self._stride

        self._prior_boxes = cxcywh2xyxy(prior_boxes.reshape(-1,4)).reshape(num_anchors, grid_y, grid_x, 4)

        return self._prior_boxes.clone()

    def logits_to_boxes(self, reg_logits:torch.Tensor) -> torch.Tensor:
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

        prior_boxes = self(grid_y, grid_x, device, dtype).reshape(-1,4) / self._stride
        prior_boxes = xyxy2cxcywh(prior_boxes).reshape(num_anchors, grid_y, grid_x, 4)

        prior_boxes[:, :, :, :2] -= 0.5

        # nA,gy,gx,4 => bs,nA,gy,gx,4
        prior_boxes = prior_boxes.repeat(batch_size, 1, 1, 1, 1)

        # bx = (sigmoid(tx) + cx) * stride
        # by = (sigmoid(ty) + cy) * stride
        # bw = exp(tw) * cw * stride
        # bh = exp(th) * ch * stride

        reg_logits = torch.sigmoid(reg_logits)

        pred_xy = reg_logits[:, :, :, :, :2] + prior_boxes[:, :, :, :, :2]
        pred_w = torch.exp(reg_logits[:, :, :, :, 2]) * self._anchors[:,0].reshape(1,-1,1,1)
        pred_h = torch.exp(reg_logits[:, :, :, :, 3]) * self._anchors[:,1].reshape(1,-1,1,1)

        pred_boxes = torch.cat([pred_xy, pred_w.unsqueeze(-1), pred_h.unsqueeze(-1)], dim=-1) * self._stride

        pred_boxes = cxcywh2xyxy(
            pred_boxes.reshape(-1,4)).reshape(
                batch_size, num_anchors, grid_y, grid_x, 4)
  
        return pred_boxes

    @staticmethod
    def generate_grids(grid_y: int, grid_x: int,
            device: str='cpu', dtype=torch.float32):
        # TODO pydoc

        # y: grid_y x grid_x
        # x: grid_y x grid_x
        y,x = torch.meshgrid(
            torch.arange(grid_y, dtype=dtype, device=device),
            torch.arange(grid_x, dtype=dtype, device=device)
        )

        # rfs: fh x fw x 2
        rfs = torch.stack([x,y], dim=-1)

        return rfs

