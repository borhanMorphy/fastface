import random
from typing import Dict, Tuple

import numpy as np

from .. import functional as F


class ColorJitter:
    """Jitters the color of the image with randomly selected values"""

    def __init__(
        self,
        p: float = 0.5,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
    ):
        super().__init__()
        self.p = p
        self.brightness_range = (-brightness, brightness)
        self.contrast_range = (-contrast, contrast)
        self.saturation_range = (-saturation, saturation)

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        assert len(img.shape) == 3, "image shape expected 3 but found: {}".format(
            len(img.shape)
        )

        if random.random() > self.p:
            value = np.random.uniform(*self.brightness_range)
            img = F.adjust_brightness(img, factor=value)

        if random.random() > self.p:
            value = np.random.uniform(*self.contrast_range)
            img = F.adjust_contrast(img, factor=value)

        if random.random() > self.p:
            value = np.random.uniform(*self.saturation_range)
            img = F.adjust_saturation(img, factor=value)

        return (img, targets)
