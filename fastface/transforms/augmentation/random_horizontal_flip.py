import random
from typing import Dict, Tuple

import numpy as np


class RandomHorizontalFlip:
    """Applies random horizontal flip for the image and updated boxes"""

    def __init__(self, p: float = 0.5):
        assert (
            p >= 0 and p <= 1.0
        ), "given `p` is not valid, must be between 0 and 1 but found: {}".format(p)
        self.p = p

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        if random.random() > self.p:
            return (img, targets)

        if len(img.shape) == 3:
            nimg = img[:, ::-1, :]
        elif len(img.shape) == 2:
            nimg = img[:, ::-1]
        else:
            raise AssertionError("image has wrong dimensions")

        target_boxes = targets.get("target_boxes")
        if (target_boxes is None) or (target_boxes.shape[0] == 0):
            return (nimg, targets)

        # x1,y1,x2,y2
        w = nimg.shape[1]
        cboxes = target_boxes.copy()
        target_boxes[:, 0] = w - cboxes[:, 2]
        target_boxes[:, 2] = w - cboxes[:, 0]

        targets["target_boxes"] = target_boxes

        return (nimg, targets)
