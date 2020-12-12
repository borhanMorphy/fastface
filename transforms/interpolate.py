import numpy as np
from typing import List,Tuple,Union,Any
import random
import math
from cv2 import cv2
from .transform import Transform

class Interpolate(Transform):
    def __init__(self, max_dim:int=640):
        super(Interpolate,self).__init__()
        self.max_dim = max_dim

    def __call__(self, img:np.ndarray,
            gt_boxes:np.ndarray=None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        h,w = img.shape[:2]

        sf = self.max_dim / max(h,w)

        nh = int(sf*h)
        nw = int(sf*w)

        if self.tracking: self.register_op({'scale_factor':sf})

        nimg = cv2.resize(img, (nw,nh), cv2.INTER_AREA)

        if isinstance(gt_boxes, type(None)): return nimg

        ngt_boxes_boxes = gt_boxes * sf
        return nimg,ngt_boxes_boxes

    def adjust(self, pred_boxes:np.ndarray, scale_factor:float=1.) -> np.ndarray:
        pred_boxes[:, :4] = pred_boxes[:, :4] / scale_factor
        return pred_boxes