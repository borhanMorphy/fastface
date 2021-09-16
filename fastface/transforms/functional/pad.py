from typing import Dict, Tuple

import numpy as np


def pad(
    img: np.ndarray, target_size: Tuple, pad_value: float = 0.0, targets: Dict = {}
):
    h, w, c = img.shape
    tw, th = target_size

    pad_left = int((tw - w) // 2) + (tw - w) % 2
    pad_right = int((tw - w) // 2)
    if w > tw:
        pad_left, pad_right = 0, 0

    pad_up = int((th - h) // 2) + (th - h) % 2
    pad_down = int((th - h) // 2)
    if h > th:
        pad_up, pad_down = 0, 0

    nimg = np.ones((th, tw, c), dtype=img.dtype) * pad_value
    nimg[pad_up : th - pad_down, pad_left : tw - pad_right] = img

    if "target_boxes" in targets:
        target_boxes = targets["target_boxes"]

        if len(target_boxes.shape) == 2 and target_boxes.shape[0] > 0:
            target_boxes[:, [0, 2]] = target_boxes[:, [0, 2]] + pad_left
            target_boxes[:, [1, 3]] = target_boxes[:, [1, 3]] + pad_up

        targets["target_boxes"] = target_boxes

    return nimg, targets
