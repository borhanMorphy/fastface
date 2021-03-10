import torch

class Anchor():
    def __init__(self, rf_stride:int, rf_start_offset:int, rf_size:int):
        self.rf_stride = rf_stride
        self.rf_start_offset = rf_start_offset
        self.rf_size = rf_size

    def __call__(self, fh:int, fw:int) -> torch.Tensor:
        """takes feature map h and w and generates rf anchors as tensor

        Args:
            fh (int): featuremap hight
            fw (int): featuremap width

        Returns:
            torch.Tensor: rf anchors as (fh x fw x 4) (xmin, ymin, xmax, ymax)
        """
        # y: fh x fw
        # x: fh x fw
        y,x = torch.meshgrid(
            torch.arange(fh),
            torch.arange(fw)
        )

        # rfs: fh x fw x 2
        rfs = torch.stack([x,y], dim=2)


        rfs *= self.rf_stride
        rfs += self.rf_start_offset

        # rfs: fh x fw x 2 as x,y
        rfs = rfs.repeat(1,1,2) # fh x fw x 2 => fh x fw x 4
        rfs[:,:,:2] = rfs[:,:,:2] - self.rf_size/2
        rfs[:,:,2:] = rfs[:,:,2:] + self.rf_size/2

        return rfs

    def logits_to_boxes(self, reg_logits:torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,fh,fw,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        fh,fw = reg_logits.shape[1:3]

        anchors = self(fh, fw).to(reg_logits.device, reg_logits.dtype)

        # anchors: fh,fw,4
        rf_normalizer = self.rf_size/2

        rf_centers = (anchors[:,:, :2] + anchors[:,:, 2:]) / 2

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