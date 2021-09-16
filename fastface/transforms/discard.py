import math
from typing import Dict, Tuple

import numpy as np


class FaceDiscarder:
    """Discard face boxes using min and max scale"""

    def __init__(self, min_face_size: int = 0, max_face_size: int = math.inf):
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:

        if "target_boxes" in targets:
            target_boxes = targets["target_boxes"]
            face_scales = (target_boxes[:, [2, 3]] - target_boxes[:, [0, 1]]).max(
                axis=1
            )
            mask = (face_scales >= self.min_face_size) & (
                face_scales <= self.max_face_size
            )
            targets["target_boxes"] = target_boxes[mask]

            if "ignore_flags" in targets:
                targets["ignore_flags"] = targets["ignore_flags"][mask]

        return (img, targets)
