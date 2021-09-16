from typing import Dict, Tuple

import numpy as np

from . import functional as F


class Padding:
    """Applies padding to image and target boxes"""

    def __init__(self, target_size: Tuple[int, int] = (640, 640), pad_value: int = 0):
        super(Padding, self).__init__()
        self.pad_value = pad_value
        self.target_size = target_size  # w,h

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        # TODO check image shape

        return F.pad(
            img, target_size=self.target_size, pad_value=self.pad_value, targets=targets
        )
