import math
import numpy as np
from typing import Tuple

class FaceDiscarder():

    def __init__(self, min_face_scale:0, max_face_scale:int=math.inf):
        self.min_face_scale = min_face_scale
        self.max_face_scale = max_face_scale

    def __call__(self, img:np.ndarray,
            gt_boxes:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:

        if isinstance(gt_boxes, type(None)): return img

        if gt_boxes.shape[0] == 0:
            return img,gt_boxes

        wh = gt_boxes[:, [2,3]] - gt_boxes[:, [0,1]]
        face_scales = wh.max(axis=1)
        accept_cond = (face_scales >= self.min_face_scale) & (face_scales <= self.max_face_scale)

        return img,gt_boxes[accept_cond, :]