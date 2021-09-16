from typing import Dict, Tuple

import numpy as np

from . import functional as F


class Interpolate:
    """Interpolates the image and boxes using target size"""

    def __init__(self, target_size: int = 640):
        super(Interpolate, self).__init__()
        self.target_size = target_size

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        assert len(img.shape) == 3, "image shape expected 3 but found: {}".format(
            len(img.shape)
        )

        nimg, targets = F.interpolate(img, self.target_size, targets=targets)

        return (nimg, targets)


class ConditionalInterpolate:
    """Interpolates the image and boxes if image height or width exceed given maximum size"""

    # TODO add to pytest

    def __init__(self, max_size: int = 640):
        super(ConditionalInterpolate, self).__init__()
        self.max_size = max_size

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        assert len(img.shape) == 3, "image shape expected 3 but found: {}".format(
            len(img.shape)
        )
        if max(img.shape) <= self.max_size:
            return (img, targets)

        nimg, targets = F.interpolate(img, self.max_size, targets=targets)

        return (nimg, targets)
