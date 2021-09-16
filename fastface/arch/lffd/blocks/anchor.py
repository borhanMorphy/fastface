import torch
import torch.nn as nn

from ....utils.box import generate_grids


class Anchor(nn.Module):
    def __init__(self, rf_stride: int, rf_start_offset: int, rf_size: int):
        super().__init__()
        self.rf_stride = rf_stride
        self.rf_start_offset = rf_start_offset
        self.rf_size = rf_size
        grids = generate_grids(1500 // rf_stride, 1500 // rf_stride)
        rfs = (grids * rf_stride + rf_start_offset).repeat(
            1, 1, 2
        )  # fh x fw x 2 => fh x fw x 4
        rfs[:, :, :2] = rfs[:, :, :2] - rf_size / 2
        rfs[:, :, 2:] = rfs[:, :, 2:] + rf_size / 2
        rf_centers = (rfs[:, :, :2] + rfs[:, :, 2:]) / 2

        # pylint: disable=not-callable
        self.register_buffer(
            "rf_normalizer", torch.tensor(rf_size / 2), persistent=False
        )
        self.register_buffer("rfs", rfs, persistent=False)
        self.register_buffer("rf_centers", rf_centers.repeat(1, 1, 2), persistent=False)
        # rfs: fh x fw x 4 as x1,y1,x2,y2

    def estimated_forward(self, imgh: int, imgw: int) -> torch.Tensor:
        """Estimates anchors using image dimensions

        Args:
            imgh (int): image height
            imgw (int): image width

        Returns:
            torch.Tensor: anchors with shape (fh x fw x 4) as xmin, ymin, xmax, ymax
        """
        fh = imgh // self.rf_stride - 1
        fw = imgw // self.rf_stride - 1
        return self.forward(fh, fw)

    def forward(self, fh: int, fw: int) -> torch.Tensor:
        """Generates anchors using featuremap dimensions

        Args:
            fh (int): featuremap hight
            fw (int): featuremap width

        Returns:
            torch.Tensor: anchors with shape (fh x fw x 4) as xmin, ymin, xmax, ymax
        """
        return self.rfs[:fh, :fw, :]
