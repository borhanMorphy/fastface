import random
import numpy as np
from typing import Tuple

class RandomHorizontalFlip():
    def __init__(self, p:float=0.5):
        assert p >= 0 and p <= 1
        self.p = p

    def __call__(self, img:np.ndarray, boxes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            return img,boxes
        
        if len(img.shape) == 3:
            nimg = img[:,::-1,:]
        elif len(img.shape) == 2:
            nimg = img[:,::-1]
        else:
            raise AssertionError("image has wrong dimensions")
        
        if boxes.shape[0] == 0:
            return nimg,boxes

        # x1,y1,x2,y2
        w = nimg.shape[1]
        cboxes = boxes.copy()
        boxes[:, 0] = w - cboxes[:, 2]
        boxes[:, 2] = w - cboxes[:, 0]

        return nimg,boxes
