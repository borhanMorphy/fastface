import numpy as np
from typing import Tuple,Union,List

class Normalize():
    def __init__(self, mean:Union[List,Tuple,float]=0, std:Union[List,Tuple,float]=1):
        self.mean = mean
        self.std = std

    def __call__(self, img:np.ndarray, boxes:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
        img -= self.mean
        img /= self.std

        return img,boxes
