from typing import List
import torch
import torch.nn as nn
from ....utils import box

class Anchor(nn.Module):

    def __init__(self, anchors: List, stride: int):
        super().__init__()
        self.stride = stride
        self.anchors = torch.tensor(anchors)

        nA = len(anchors)

        grids = box.generate_grids(1500//stride, 1500//stride)
        # grids fh x fw x 2 as x1,y1

        centers = (grids + 0.5).repeat(nA, 1, 1, 1) # cx,cy
        centers *= stride
        # centers: nA x fh x fw x 2

        wh = torch.ones_like(centers) * self.anchors.view(nA, 1, 1, 2)
 
        priors = torch.cat([centers, wh], dim=-1)

        self.register_buffer("priors", priors)

    def forward(self, fh: int, fw: int) -> torch.Tensor:
        """Generates anchors using featuremap dimensions

        Args:
            fh (int): featuremap hight
            fw (int): featuremap width

        Returns:
            torch.Tensor: anchors with shape (na x fh x fw x 4) as cx, cy, w, h
        """
        return self.priors[:, :fh, :fw, :]
