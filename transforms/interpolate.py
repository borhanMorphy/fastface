import numpy as np
from typing import List,Tuple
import random
import math
from cv2 import cv2

class Interpolate():
    def __init__(self, max_dim:int=640):
        self.max_dim = max_dim

    def __call__(self, img:np.ndarray, boxes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h,w = img.shape[:2]

        sf = self.max_dim / max(h,w)

        nh = int(sf*h)
        nw = int(sf*w)

        nimg = cv2.resize(img, (nw,nh), cv2.INTER_AREA)
        nboxes = boxes * sf
        return nimg,nboxes

