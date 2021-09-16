from typing import Dict

import numpy as np
from PIL import Image


def interpolate(img: np.ndarray, target_size: int, targets: Dict = {}):
    h, w = img.shape[:2]

    sf = target_size / max(h, w)

    nh = int(sf * h)
    nw = int(sf * w)

    nimg = np.array(Image.fromarray(img).resize((nw, nh)), dtype=img.dtype)

    if "target_boxes" in targets:
        targets["target_boxes"] *= sf

    return nimg, targets
