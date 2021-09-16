from typing import Dict, Tuple

import numpy as np

from . import functional as F


class Rotate:
    """Rotates the image and boxes clockwise using given degree"""

    def __init__(self, degree: float = 0):
        super().__init__()
        self.degree = degree

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        assert len(img.shape) == 3, "image shape expected 3 but found: {}".format(
            len(img.shape)
        )

        nimg, targets = F.rotate(img, self.degree, targets=targets)

        return (nimg, targets)
