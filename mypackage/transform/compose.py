from typing import Tuple
import numpy as np

class Compose():
    def __init__(self, *ts):
        self.ts = ts
        self.__ops_cache = []

    def __call__(self, img:np.ndarray, gt_boxes:np.ndarray=None):
        for t in self.ts:
            if isinstance(gt_boxes, type(None)):
                img = t(img)
            else:
                img,gt_boxes = t(img,gt_boxes)
            self.__ops_cache.append(t)

        if isinstance(gt_boxes, type(None)): return img

        return img,gt_boxes

    def enable_tracking(self):
        for t in self.ts:
            if not hasattr(t,"enable_tracking"): continue
            t.enable_tracking()

    def disable_tracking(self):
        for t in self.ts:
            if not hasattr(t,"disable_tracking"): continue
            t.disable_tracking()

    def flush(self):
        for t in self.ts:
            if not hasattr(t,"flush"): continue
            t.flush()

    def adjust(self, pred_boxes:np.ndarray) -> np.ndarray:
        """prediction boxes as np.ndarray(N,5)

        Args:
            pred_boxes (np.ndarray): as x1,y1,x2,y2,score

        Returns:
            np.ndarray: adjusted boxes
        """
        adjusted_boxes = pred_boxes.copy()
        for t in reversed(self.__ops_cache):
            if not hasattr(t,'adjust'): continue
            adjusted_boxes = t._adjust(adjusted_boxes)
        self.__ops_cache = []
        return adjusted_boxes