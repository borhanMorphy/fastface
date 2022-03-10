from typing import List, Tuple
import random

from cv2 import cv2
from albumentations.core.transforms_interface import DualTransform
import albumentations.augmentations.geometric.functional as F_geometric
import albumentations.augmentations.crops as F_crop

class RandomScaleSample(DualTransform):
    """augmentation for scale-based sampling, defined in the paper
    """

    def __init__(
        self,
        scales: List[Tuple[int, int]],
        ref_size: int = 640,
        always_apply: bool = False,
        p: float = 1,
    ):
        super(RandomScaleSample, self).__init__(always_apply, p)
        self.ref_size = ref_size
        self.scales = scales
        self.value = 0

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]

        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            return {
                "dx": 0,
                "dy": 0,
                "cs": 1.0,
                "ts": 1.0,
            }

        # select one face
        selected_face_idx = random.randint(0, len(params["bboxes"]) - 1)

        selected_face_scale_idx = random.choice(list(range(len(self.scales))))
        min_scale, max_scale = self.scales[selected_face_scale_idx]

        ts = random.uniform(min_scale, max_scale)
        
        # check if boxes are normalized
        xmin, ymin, xmax, ymax, *_ = params["bboxes"][selected_face_idx]

        dx = 0.5 - (xmax + xmin) / 2
        dy = 0.5 - (ymax + ymin) / 2

        cs = max((xmax - xmin) * img_w, (ymax - ymin) * img_h)

        return {"dx": dx, "dy": dy, "cs": cs, "ts": ts}

    def apply(self, img, dx=0, dy=0, cs=1, ts=1, rows=None, cols=None, **params):
        # shift to selected bbox
        img = F_geometric.shift_scale_rotate(img, 0, 1, dx, dy, border_mode=cv2.BORDER_CONSTANT, value=0)

        if ts > cs:
            target_size = int(self.ref_size * (cs / ts))
            crop_w = min(target_size, cols)
            crop_h = min(target_size, rows)
            # apply center crop
            img = F_crop.center_crop(img, crop_h, crop_w)

        img_h, img_w = img.shape[:2]

        return F_geometric.resize(img, height=int((ts/cs) * img_h), width=int((ts/cs) * img_w))

    def apply_to_bbox(self, bbox, dx=0, dy=0, cs=1, ts=1, rows=None, cols=None, **params):

        # shift bbox
        bbox = F_geometric.bbox_shift_scale_rotate(bbox, 0, 1, dx, dy, rows, cols)

        if ts > cs:
            target_size = int(self.ref_size * (cs / ts))
            crop_w = min(target_size, cols)
            crop_h = min(target_size, rows)
            # apply center crop
            bbox = F_crop.bbox_center_crop(bbox, crop_h, crop_w, rows, cols)

        # bbox coords are scale invariant
        return bbox

    def get_transform_init_args_names(self):
        return (
            "ref_size",
            "scales",
        )

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]
