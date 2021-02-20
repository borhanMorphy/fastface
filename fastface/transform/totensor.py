import torch
import numpy as np
from typing import Tuple

class ToTensor():
    """Transforms numpy image and boxes to torch tensor
    """
    def __call__(self, img:np.ndarray,
            gt_boxes:np.ndarray=None) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(img).float().permute(2,0,1)

        if isinstance(gt_boxes, type(None)): return img

        gt_boxes = torch.from_numpy(gt_boxes)
        return img,gt_boxes