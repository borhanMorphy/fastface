import torch

class AnchorGenerator():
    def __init__(self):
        pass # TODO

    def __call__(self) -> torch.Tensor:
        pass # TODO

    def logits_to_boxes(self, reg_logits:torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,fh,fw,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        pass # TODO