from typing import Tuple
from albumentations import HorizontalFlip

class KeypointOrderAwareHFlip(HorizontalFlip):
    """horizontal flip with preserving the key points order
    Args:
        flip_order (Tuple): order per keypoint after flip operation eg: (1, 0, 2, 4, 3)
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, flip_order: Tuple, p: float = 0.5):
        super().__init__(p=p)
        self.flip_order = flip_order
        self.num_of_landmarks = len(flip_order)

    def apply_to_keypoints(self, keypoints, **params):
        all_keypoints = list()
        person_keypoints = list()
        assert len(keypoints) % self.num_of_landmarks == 0, "length of keypoints must be divisible by number of landmarks per face"

        for idx, keypoint in enumerate(keypoints):
            person_keypoints.append(
                self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:])
            )
            if idx % self.num_of_landmarks == self.num_of_landmarks-1:
                # reorder
                person_keypoints = [[*person_keypoints[order_idx][:-1], person_keypoints[i][-1]] for i, order_idx in enumerate(self.flip_order)]
                # append
                all_keypoints += person_keypoints
                # refresh
                person_keypoints = list()
        assert len(person_keypoints) == 0, "flip order length is not matches with the number of landmarks per person"
        return all_keypoints
