import torch
import torch.nn as nn
from ...utils.box import generate_grids

class Anchor(nn.Module):

    def __init__(self, rf_stride: int, rf_start_offset: int, rf_size: int):
        super().__init__()
        self.rf_stride = rf_stride
        self.rf_start_offset = rf_start_offset
        self.rf_size = rf_size

    @torch.jit.unused
    def estimated_forward(self, imgh: int, imgw: int) -> torch.Tensor:
        """Estimates anchors using image dimensions

        Args:
            imgh (int): image height
            imgw (int): image width

        Returns:
            torch.Tensor: anchors with shape (fh x fw x 4) as xmin, ymin, xmax, ymax
        """
        fh = imgh // self.rf_stride
        fw = imgw // self.rf_stride
        return self.forward(fh, fw)

    def forward(self, fh: int, fw: int) -> torch.Tensor:
        """Generates anchors using featuremap dimensions

        Args:
            fh (int): featuremap hight
            fw (int): featuremap width

        Returns:
            torch.Tensor: anchors with shape (fh x fw x 4) as xmin, ymin, xmax, ymax
        """
        grids = generate_grids(fh, fw)
        rfs = grids*self.rf_stride
        rfs = rfs+self.rf_start_offset
        rfs = rfs.repeat(1, 1, 2) # fh x fw x 2 => fh x fw x 4
        rfs[:, :, :2] = rfs[:, :, :2] - self.rf_size/2
        rfs[:, :, 2:] = rfs[:, :, 2:] + self.rf_size/2
        return rfs

    def logits_to_boxes(self, reg_logits:torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,fh,fw,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        _, fh, fw, _ = reg_logits.shape

        rfs = self.forward(fh, fw).to(reg_logits.device)

        # rfs: fh,fw,4
        rf_normalizer = self.rf_size/2

        rf_centers = (rfs[:,:, :2] + rfs[:,:, 2:]) / 2

        pred_boxes = reg_logits.clone()

        pred_boxes[:, :, :, 0] = rf_centers[:, :, 0] - (rf_normalizer*reg_logits[:, :, :, 0])
        pred_boxes[:, :, :, 1] = rf_centers[:, :, 1] - (rf_normalizer*reg_logits[:, :, :, 1])
        pred_boxes[:, :, :, 2] = rf_centers[:, :, 0] - (rf_normalizer*reg_logits[:, :, :, 2])
        pred_boxes[:, :, :, 3] = rf_centers[:, :, 1] - (rf_normalizer*reg_logits[:, :, :, 3])

        pred_boxes[:, :, :, 0] = torch.clamp(pred_boxes[:, :, :, 0], 0, fw*self.rf_stride)
        pred_boxes[:, :, :, 1] = torch.clamp(pred_boxes[:, :, :, 1], 0, fh*self.rf_stride)
        pred_boxes[:, :, :, 2] = torch.clamp(pred_boxes[:, :, :, 2], 0, fw*self.rf_stride)
        pred_boxes[:, :, :, 3] = torch.clamp(pred_boxes[:, :, :, 3], 0, fh*self.rf_stride)

        return pred_boxes

