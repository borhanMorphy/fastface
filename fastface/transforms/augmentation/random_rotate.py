import random
from typing import Dict, Tuple

import numpy as np

from .. import functional as F


class RandomRotate:
    """Rotates the image and boxes clockwise with randomly selected value"""

    def __init__(self, p: float = 0.5, degree_range: float = 0):
        super().__init__()
        self.p = p
        self.degree_range = degree_range

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        assert len(img.shape) == 3, "image shape expected 3 but found: {}".format(
            len(img.shape)
        )

        if random.random() > self.p:
            return (img, targets)

        degree = np.random.uniform(low=-self.degree_range, high=self.degree_range)

        nimg, targets = F.rotate(img, degree, targets=targets)

        return (nimg, targets)
