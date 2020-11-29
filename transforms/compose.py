from typing import Tuple
import numpy as np

class Compose():
    def __init__(self, *ts):
        self.ts = ts

    def __call__(self, img:np.ndarray, gt_boxes:np.ndarray=None):
        for t in self.ts:
            if isinstance(gt_boxes, type(None)):
                img = t(img)
            else:
                img,gt_boxes = t(img,gt_boxes)

        if isinstance(gt_boxes, type(None)): return img

        return img,gt_boxes