import numpy as np
from typing import List


class LFFDRandomSample():
    def __init__(self, scales:List[List]):
        self.selection_prob = 1/len(scales)
        self.scales = np.array(scales, dtype=np.float32) # N,2

    def __call__(self, img:np.ndarray, boxes:np.ndarray):
        pass