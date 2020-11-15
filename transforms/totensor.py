import torch
import numpy as np
from typing import Tuple

class ToTensor():
    def __call__(self, img:np.ndarray,
            boxes:np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(img).float().permute(2,0,1)
        boxes = torch.from_numpy(boxes)
        return img,boxes