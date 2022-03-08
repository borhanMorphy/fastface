from typing import Tuple
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
        # rf centers start at grid + rf_start_offset
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
        # rfs: fh x fw x 4 as x1, y1, x2, y2
        self.register_buffer("rf_centers", rf_centers, persistent=False)
        # rf_centers: fh x fw x 2

    def estimate_fmap(self, imgh: int, imgw: int) -> Tuple[int, int]:
        """Estimates fmap shape by looking at input image's height and width

        Args:
            imgh (int): input image height
            imgw (int): input image width

        Returns:
            Tuple[int, int]: fmap shape as fh, fw
        """
        fh = imgh // self.rf_stride - 1
        fw = imgw // self.rf_stride - 1
        return fh, fw

    def estimated_forward(self, imgh: int, imgw: int) -> torch.Tensor:
        """Estimates anchors using image dimensions

        Args:
            imgh (int): image height
            imgw (int): image width

        Returns:
            torch.Tensor: anchors with shape (fh x fw x 4) as xmin, ymin, xmax, ymax
        """
        fh, fw = self.estimate_fmap(imgh, imgw)
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
