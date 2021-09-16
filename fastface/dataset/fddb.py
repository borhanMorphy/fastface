import math
import os
from typing import List

import numpy as np

from ..utils.cache import get_data_cache_dir
from .base import BaseDataset


def _ellipse2box(major_r, minor_r, angle, center_x, center_y):
    tan_t = -(minor_r / major_r) * math.tan(angle)
    t = math.atan(tan_t)
    x1 = center_x + (
        major_r * math.cos(t) * math.cos(angle)
        - minor_r * math.sin(t) * math.sin(angle)
    )
    x2 = center_x + (
        major_r * math.cos(t + math.pi) * math.cos(angle)
        - minor_r * math.sin(t + math.pi) * math.sin(angle)
    )
    x_max = max(x1, x2)
    x_min = min(x1, x2)

    if math.tan(angle) != 0:
        tan_t = (minor_r / major_r) * (1 / math.tan(angle))
    else:
        tan_t = (minor_r / major_r) * (1 / (math.tan(angle) + 0.0001))
    t = math.atan(tan_t)
    y1 = center_y + (
        minor_r * math.sin(t) * math.cos(angle)
        + major_r * math.cos(t) * math.sin(angle)
    )
    y2 = center_y + (
        minor_r * math.sin(t + math.pi) * math.cos(angle)
        + major_r * math.cos(t + math.pi) * math.sin(angle)
    )
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
            elif line != "":
                # indicates box
                # 123.583300 85.549500 1.265839 269.693400 161.781200  1
                major_r, minor_r, angle, cx, cy, _ = [
                    float(point) for point in line.split(" ") if point != ""
                ]
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

    __URLS__ = {
        "fddb-images": {
            "adapter": "http",
            "check": {
                "2002": "ffd8ac86d9f407ac415cfe4dd2421407",
                "2003": "6356cbce76b26a92fc9788b221f5e5bb",
            },
            "kwargs": {
                "url": "http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz",
                "extract": True,
            },
        },
        "fddb-annotations": {
            "adapter": "http",
            "check": {"FDDB-folds": "694de7a9144611e2353b7055819026e3"},
            "kwargs": {
                "url": "http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz",
                "extract": True,
            },
        },
    }

    __phases__ = ("train", "val", "test")
    __folds__ = tuple((i + 1 for i in range(10)))
    __splits__ = ((1, 2, 4, 5, 7, 9, 10), (3, 6, 8), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    def __init__(
        self,
        source_dir: str = None,
        phase: str = None,
        folds: List[int] = None,
        transforms=None,
        **kwargs
    ):

        source_dir = (
            get_data_cache_dir(suffix="fddb") if source_dir is None else source_dir
        )

        # check if download
        self.download(self.__URLS__, source_dir)

        assert os.path.exists(
            source_dir
        ), "given source directory for fddb is not exist at {}".format(source_dir)
        assert (
            phase is None or phase in FDDBDataset.__phases__
        ), "given phase {} is \
            not valid, must be one of: {}".format(
            phase, self.__phases__
        )

        if phase is None:
            folds = list(self.__folds__) if folds is None else folds
        else:
            # TODO log here if `phase` is not None, folds argument will be ignored
            folds = self.__splits__[self.__phases__.index(phase)]

        ids = []
        targets = []
        for fold_idx in folds:
            assert (
                fold_idx in self.__folds__
            ), "given fold {} is not in the fold list".format(fold_idx)
            raw_ids, raw_targets = _load_single_annotation_fold(source_dir, fold_idx)
            ids += raw_ids
            # TODO each targets must be dict
            for target in raw_targets:
                targets.append({"target_boxes": target.astype(np.float32)})
            del raw_targets
        super().__init__(ids, targets, transforms=transforms, **kwargs)
