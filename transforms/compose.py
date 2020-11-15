from typing import Tuple
import numpy as np

class Compose():
    def __init__(self, *ts):
        self.ts = ts

    def __call__(self, img:np.ndarray, gt_boxes:np.ndarray) -> Tuple:
        for t in self.ts:
            img,gt_boxes = t(img,gt_boxes)
        return img,gt_boxes