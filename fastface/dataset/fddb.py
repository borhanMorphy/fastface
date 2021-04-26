__all__ = ["FDDBDataset"]

import os
from typing import List
import math
import numpy as np

from .base import BaseDataset

def _ellipse2box(major_r, minor_r, angle, center_x, center_y):
    tan_t = -(minor_r/major_r)*math.tan(angle)
    t = math.atan(tan_t)
    x1 = center_x + (major_r*math.cos(t)*math.cos(angle) - minor_r*math.sin(t)*math.sin(angle))
    x2 = center_x + (major_r*math.cos(t+math.pi)*math.cos(angle) - minor_r*math.sin(t+math.pi)*math.sin(angle))
    x_max = max(x1,x2)
    x_min = min(x1,x2)

    if math.tan(angle) != 0:
        tan_t = (minor_r/major_r)*(1/math.tan(angle))
    else:
        tan_t = (minor_r/major_r)*(1/(math.tan(angle)+0.0001))
    t = math.atan(tan_t)
    y1 = center_y + (minor_r*math.sin(t)*math.cos(angle) + major_r*math.cos(t)*math.sin(angle))
    y2 = center_y + (minor_r*math.sin(t+math.pi)*math.cos(angle) + major_r*math.cos(t+math.pi)*math.sin(angle))
    y_max = max(y1, y2)
    y_min = min(y1, y2)

    return x_min, y_min, x_max, y_max

def _load_single_annotation_fold(source_path: str, fold_idx: int):
    # source_path/FDDB-fold-{:02d}-ellipseList.txt
    # TODO check fold idx range

    fold_file_name = "FDDB-fold-{:02d}-ellipseList.txt".format(fold_idx)
    fold_prefix = "FDDB-folds"

    img_file_path = os.path.join(source_path, "{}.jpg")

    fold_file_path = os.path.join(source_path, fold_prefix, fold_file_name)
    ids = []
    targets = []
    boxes = []

    with open(fold_file_path, "r") as foo:
        for line in foo.read().split("\n"):
            if os.path.isfile(img_file_path.format(line)):
                # indicates img file path
                if len(boxes) > 0:
                    boxes = np.array(boxes)
                    targets.append(boxes)

                ids.append(img_file_path.format(line))
                boxes = []
            elif line.isnumeric():
                # indicates number of face line
                pass
            elif line != '':
                # indicates box
                # 123.583300 85.549500 1.265839 269.693400 161.781200  1
                major_r, minor_r, angle, cx, cy, _ = [float(l) for l in line.split(" ") if l != '']
                box = _ellipse2box(major_r, minor_r, angle, cx, cy)
                boxes.append(box)
        if len(boxes) > 0:
            boxes = np.array(boxes)
            targets.append(boxes)

    return ids, targets

class FDDBDataset(BaseDataset):
    """FDDB fastface.dataset.BaseDataset Instance

    paper: http://vis-www.cs.umass.edu/fddb/fddb.pdf
    specs:
        - total number of images: 2845
        - total number of faces: 5171

    """

    __phases__ = ("train", "val", "custom")
    __folds__ = tuple((i+1 for i in range(10)))
    __splits__ = ((1, 2, 4, 5, 7, 9, 10), (3, 6, 8))

    def __init__(self, source_dir: str, phase: str = "custom",
            folds: List[int] = None, transforms=None, **kwargs):
        # TODO make use of `phase`
        assert phase in FDDBDataset.__phases__, f"given phase {phase} is not valid, must be one of: {self.__phases__}"

        folds = list(self.__folds__) if folds is None else folds

        if phase != "custom":
            folds = self.__splits__[self.__phases__.index(phase)]

        ids = []
        targets = []
        for fold_idx in folds:
            assert fold_idx in self.__folds__, f"given fold {fold_idx} is not in the fold list"
            raw_ids, raw_targets = _load_single_annotation_fold(source_dir, fold_idx)
            ids += raw_ids
            # TODO each targets must be dict
            for target in raw_targets:
                targets.append({
                    "target_boxes": target.astype(np.float32)
                })
            del raw_targets
        super().__init__(ids, targets, transforms=transforms, **kwargs)