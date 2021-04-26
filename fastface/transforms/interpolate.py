import numpy as np
from typing import Tuple, Dict
from PIL import Image

class Interpolate():
    """Interpolates the image and boxes using maximum dimension"""

    def __init__(self, max_dim: int = 640):
        super(Interpolate, self).__init__()
        self.max_dim = max_dim

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        assert len(img.shape) == 3, "image shape expected 3 but found: {}".format(len(img.shape))
        h, w = img.shape[:2]

        sf = self.max_dim / max(h, w)

        nh = int(sf*h)
        nw = int(sf*w)

        nimg = np.array(Image.fromarray(img).resize((nw, nh)), dtype=img.dtype)

        if "target_boxes" in targets:
            targets["target_boxes"] *= sf

        return (nimg, targets)