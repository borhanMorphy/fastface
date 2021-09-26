from typing import Dict, Tuple

import numpy as np


class Compose:
    """Compose given transforms"""

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, img: np.ndarray, targets: Dict = None) -> Tuple[np.ndarray, Dict]:
        targets = dict() if targets is None else targets
        # TODO add logger
        for transform in self.transforms:
            img, targets = transform(img, targets=targets)

        return (img, targets)
