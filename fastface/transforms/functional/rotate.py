from typing import Dict

import numpy as np
from PIL import Image

from ...utils.geo import get_rotation_matrix


def rotate(img: np.ndarray, degree: int, targets: Dict = {}) -> np.ndarray:
    # clockwise rotation

    h, w = img.shape[:2]
    cx = w // 2
    cy = h // 2

    nimg = np.array(Image.fromarray(img).rotate((360 - degree), center=(cx, cy)))

    if "target_boxes" in targets:
        r = get_rotation_matrix(degree)
        N = targets["target_boxes"].shape[0]
        if N == 0:
            return nimg, targets

        coords = np.empty((N, 4, 2), dtype=targets["target_boxes"].dtype)

        # x1,y1
        coords[:, 0, :] = targets["target_boxes"][:, [0, 1]]
        # x1,y2
        coords[:, 1, :] = targets["target_boxes"][:, [0, 3]]
        # x2,y1
        coords[:, 2, :] = targets["target_boxes"][:, [2, 1]]
        # x2,y2
        coords[:, 3, :] = targets["target_boxes"][:, [2, 3]]

        # convert to regular coodinate space
        coords[..., [0]] = w - coords[..., [0]]
        coords[..., [1]] = h - coords[..., [1]]

        # centerize the coordinates
        coords[..., [0]] -= cx
        coords[..., [1]] -= cy

        # apply clockwise rotation
        coords = np.matmul(coords, r)

        # revert centerization
        coords[..., [0]] += cx
        coords[..., [1]] += cy

        # re-convert to image coordinate space
        coords[..., [0]] = w - coords[..., [0]]
        coords[..., [1]] = h - coords[..., [1]]

        # create the box
        x1 = coords[..., 0].min(axis=1)
        y1 = coords[..., 1].min(axis=1)
        x2 = coords[..., 0].max(axis=1)
        y2 = coords[..., 1].max(axis=1)

        # stack
        n_boxes = np.stack([x1, y1, x2, y2], axis=1)

        # clip
        n_boxes[:, [0, 2]] = n_boxes[:, [0, 2]].clip(min=0, max=w - 1)
        n_boxes[:, [1, 3]] = n_boxes[:, [1, 3]].clip(min=0, max=h - 1)

        targets["target_boxes"] = n_boxes

    return nimg, targets
