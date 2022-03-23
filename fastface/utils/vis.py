from typing import Dict, List, Tuple

import numpy as np
from cv2 import cv2


def draw_faces(
    data: Dict, box_color=(0, 255, 0), keypoint_color=(0, 255, 0)
) -> np.ndarray:
    assert "image" in data
    keypoint_exists = "keypoints" in data

    img = data["image"].copy()
    bboxes = data["bboxes"]

    if len(bboxes) == 0:
        return img

    num_of_keypoints = -1
    if keypoint_exists:
        num_of_keypoints = len(data["keypoints"]) // len(bboxes)

    for person_idx in range(len(bboxes)):
        box = bboxes[person_idx]
        keypoints = None
        if keypoint_exists:
            keypoints = data["keypoints"][
                person_idx * num_of_keypoints : (person_idx + 1) * num_of_keypoints
            ]

        img = draw_face(
            img,
            box,
            keypoints=keypoints,
            box_color=box_color,
            keypoint_color=keypoint_color,
        )

    return img


def draw_face(
    img,
    box: List[float],
    keypoints: List[Tuple[float, float]] = None,
    box_color=(0, 255, 0),
    keypoint_color=(0, 255, 0),
):

    x1, y1, x2, y2 = [int(b) for b in box]
    img = cv2.rectangle(img, (x1, y1), (x2, y2), box_color)

    key_radius = max((y2 - y1), (x2 - x1)) // 20

    for kx, ky in keypoints or list():
        kx = int(kx)
        ky = int(ky)
        img = cv2.circle(img, (kx, ky), key_radius, keypoint_color)

    return img
