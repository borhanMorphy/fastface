from typing import Tuple, Dict
import numpy as np

class Padding():
    """Applies padding to image and target boxes"""

    def __init__(self, target_size: Tuple[int, int] = (640, 640), pad_value: int = 0):
        super(Padding, self).__init__()
        self.pad_value = pad_value
        self.target_size = target_size # w,h

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        # TODO check image shape

        h, w, c = img.shape
        tw, th = self.target_size

        pad_left = int((tw - w) // 2) + (tw-w) % 2
        pad_right = int((tw - w) // 2)
        if w > tw:
            pad_left, pad_right = 0, 0

        pad_up = int((th - h) // 2) + (th-h) % 2
        pad_down = int((th - h) // 2)
        if h > th:
            pad_up, pad_down = 0,0

        nimg = np.ones((th, tw, c), dtype=img.dtype) * self.pad_value
        nimg[pad_up:th-pad_down, pad_left:tw-pad_right] = img

        if "target_boxes" in targets:
            target_boxes = targets["target_boxes"]

            if len(target_boxes.shape) == 2 and target_boxes.shape[0] > 0:
                target_boxes[:, [0,2]] = target_boxes[:, [0,2]] + pad_left
                target_boxes[:, [1,3]] = target_boxes[:, [1,3]] + pad_up

            targets["target_boxes"] = target_boxes

        return (nimg, targets)