from typing import Dict, List, Tuple, Union

import numpy as np


class Normalize:
    """Normalizes the image with given `mean` and `std`. (img - mean) / std"""

    def __init__(
        self, mean: Union[List, Tuple, float] = 0, std: Union[List, Tuple, float] = 1
    ):
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        img = img.astype(np.float32)
        img = img - self.mean
        img = img / self.std

        return (img, targets)
