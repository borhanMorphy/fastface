import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from ..interpolate import Interpolate
from ..pad import Padding


class LFFDRandomSample:
    """Applies augmantation defined in the LFFD paper"""

    def __init__(
        self,
        scales: List[Tuple[int, int]],
        target_size: Tuple[int, int] = (640, 640),
        p: float = 0.5,
    ):
        assert (
            p >= 0 and p <= 1.0
        ), "given `p` is not valid, must be between 0 and 1 but found: {}".format(p)
        self.scales = scales
        self.target_size = target_size  # W,H
        self.padding = Padding(target_size=target_size, pad_value=0)
        self.interpolate = Interpolate(target_size=target_size[0])
        self.p = p

    def __call__(self, img: np.ndarray, targets: Dict = {}) -> Tuple[np.ndarray, Dict]:
        """Randomly samples faces using given scales. All scales represents branches and
        for each branch selection probability is same.

        Args:
            img (np.ndarray): H,W,C
            targets (Dict, Optional): contains targets

        Returns:
            Tuple[np.ndarray, Dict]: transformed image and transformed targets
        """
        target_boxes = targets.get("target_boxes")

        if (
            (target_boxes is None)
            or (target_boxes.shape[0] == 0)
            or (random.random() > self.p)
        ):
            img, targets = self.interpolate(img, targets=targets)
            img, targets = self.padding(img, targets=targets)
            return (img, targets)

        num_faces = target_boxes.shape[0]

        # select one face
        selected_face_idx = random.randint(0, num_faces - 1)

        selected_face_scale_idx = random.choice(list(range(len(self.scales))))
        min_scale, max_scale = self.scales[selected_face_scale_idx]

        scale_size = random.uniform(min_scale, max_scale)

        x1, y1, x2, y2 = target_boxes[selected_face_idx].astype(np.int32)

        face_scale = max(y2 - y1, x2 - x1)
        h, w = img.shape[:2]

        sf = scale_size / face_scale

        aboxes = target_boxes * sf
        sx1, sy1, sx2, sy2 = aboxes[selected_face_idx].astype(np.int32)

        offset_w_1 = (self.target_size[0] - (sx2 - sx1)) // 2
        offset_w_2 = offset_w_1 + (self.target_size[0] - (sx2 - sx1)) % 2

        offset_w_1 //= sf
        offset_w_2 //= sf

        offset_h_1 = (self.target_size[1] - (sy2 - sy1)) // 2
        offset_h_2 = offset_h_1 + (self.target_size[1] - (sy2 - sy1)) % 2

        offset_h_1 //= sf
        offset_h_2 //= sf

        offset_w_1 = int(min(x1, offset_w_1))
        offset_w_2 = int(min(w - x2, offset_w_2))

        offset_h_1 = int(min(y1, offset_h_1))
        offset_h_2 = int(min(h - y2, offset_h_2))

        # select faces that center's lie between cropped region
        low_h, high_h = y1 - offset_h_1, y2 + offset_h_2
        low_w, high_w = x1 - offset_w_1, x2 + offset_w_2
        cboxes_x = (target_boxes[:, 0] + target_boxes[:, 2]) // 2
        cboxes_y = (target_boxes[:, 1] + target_boxes[:, 3]) // 2

        # TODO handle here
        center_mask = np.bitwise_and(
            np.bitwise_and(cboxes_x > low_w, cboxes_x < high_w),
            np.bitwise_and(cboxes_y > low_h, cboxes_y < high_h),
        )

        aimg = img[y1 - offset_h_1 : y2 + offset_h_2, x1 - offset_w_1 : x2 + offset_w_2]

        # TODO control this line
        aimg = np.array(
            Image.fromarray(aimg).resize(
                (int(aimg.shape[1] * sf), int(aimg.shape[0] * sf))
            )
        )

        aimg = aimg[: self.target_size[1], : self.target_size[0]]

        target_boxes[:, [0, 2]] = target_boxes[:, [0, 2]] - (x1 - offset_w_1)
        target_boxes[:, [1, 3]] = target_boxes[:, [1, 3]] - (y1 - offset_h_1)
        target_boxes *= sf

        x1, y1, x2, y2 = target_boxes[selected_face_idx].astype(np.int32)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        img = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        tcx = img.shape[1] // 2
        tcy = img.shape[0] // 2

        offset_x = int(tcx - cx)
        offset_y = int(tcy - cy)

        if offset_x >= 0:
            # pad left
            left_index_x = offset_x
            right_index_x = offset_x + aimg.shape[1]
        else:
            # pad_right
            left_index_x = 0
            right_index_x = aimg.shape[1]
        if offset_y >= 0:
            # pad up
            up_index_y = offset_y
            down_index_y = offset_y + aimg.shape[0]
        else:
            # pad down
            up_index_y = 0
            down_index_y = aimg.shape[0]

        target_h, target_w = img[
            up_index_y:down_index_y, left_index_x:right_index_x
        ].shape[:2]
        source_h, source_w = aimg.shape[:2]

        up_index_y = up_index_y + target_h - source_h
        down_index_y = down_index_y + target_h - source_h
        left_index_x = left_index_x + target_w - source_w
        right_index_x = right_index_x + target_w - source_w

        img[up_index_y:down_index_y, left_index_x:right_index_x] = aimg

        target_boxes[:, [0, 2]] += left_index_x
        target_boxes[:, [1, 3]] += up_index_y

        target_boxes[:, 0] = target_boxes[:, 0].clip(0, self.target_size[0])
        target_boxes[:, 1] = target_boxes[:, 1].clip(0, self.target_size[1])
        target_boxes[:, 2] = target_boxes[:, 2].clip(0, self.target_size[0])
        target_boxes[:, 3] = target_boxes[:, 3].clip(0, self.target_size[1])
        targets["target_boxes"] = target_boxes[center_mask, :]

        return (img, targets)
