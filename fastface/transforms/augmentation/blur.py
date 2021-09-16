import random
from typing import Dict, Tuple

import numpy as np

from ...utils import kernel


class RandomGaussianBlur:
    """TODO"""

    def __init__(self, p: float = 0.5, kernel_size: int = 15, sigma: float = 5):
        super().__init__()
        self.p = p
        self.kernel = kernel.get_gaussian_kernel(kernel_size, sigma=sigma)

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        assert len(img.shape) == 3, "image shape expected 3 but found: {}".format(
            len(img.shape)
        )

        if random.random() > self.p:
            return kernel.apply_conv2d(img, self.kernel), targets

        return (img, targets)
