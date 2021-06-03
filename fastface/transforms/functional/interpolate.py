from typing import Dict
from PIL import Image
import numpy as np

def interpolate(img: np.ndarray, target_size: int, targets: Dict = {}):
    h, w = img.shape[:2]

    sf = target_size / max(h, w)

    nh = int(sf*h)
    nw = int(sf*w)

    nimg = np.array(Image.fromarray(img).resize((nw, nh)), dtype=img.dtype)

    if "target_boxes" in targets:
        targets["target_boxes"] *= sf

    return nimg, targets
