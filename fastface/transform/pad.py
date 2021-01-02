from typing import Tuple
import numpy as np
from .transform import Transform

class Padding(Transform):

    def __init__(self, target_size:Tuple[int,int]=(640,640), pad_value:int=0):
        super(Padding,self).__init__()
        self.pad_value = pad_value
        self.target_size = target_size # w,h

    def __call__(self, img:np.ndarray,
            gt_boxes:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:

        h,w,c = img.shape
        tw,th = self.target_size

        pad_left = int((tw - w) // 2) + (tw-w) % 2
        pad_right = int((tw - w) // 2)
        if w > tw: pad_left,pad_right = 0,0

        pad_up = int((th - h) // 2) + (th-h) % 2
        pad_down = int((th - h) // 2)
        if h > th: pad_up,pad_down = 0,0

        nimg = np.ones((th,tw,c), dtype=img.dtype) * self.pad_value
        nimg[pad_up:th-pad_down, pad_left:tw-pad_right] = img

        if self.tracking: self.register_op({'pad_left':pad_left, 'pad_up':pad_up})

        if isinstance(gt_boxes, type(None)): return nimg

        ngt_boxes = gt_boxes.copy()
        if len(gt_boxes.shape) == 2 and gt_boxes.shape[0] > 0:
            ngt_boxes[:, [0,2]] = gt_boxes[:, [0,2]] + pad_left
            ngt_boxes[:, [1,3]] = gt_boxes[:, [1,3]] + pad_up
        return nimg, ngt_boxes

    def adjust(self, pred_boxes:np.ndarray,
            pad_left:int=0, pad_up:int=0) -> np.ndarray:
        pred_boxes[:, [0,2]] -= pad_left
        pred_boxes[:, [1,3]] -= pad_up
        return pred_boxes