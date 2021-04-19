from torch.utils.data import Dataset
import os
from typing import List,Tuple
import numpy as np
import imageio
import torch
import math


def ellipse2box(major_r, minor_r, angle, center_x, center_y):
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
    y_max = max(y1,y2)
    y_min = min(y1,y2)

    return x_min, y_min, x_max, y_max


def load_single_annotation_fold(source_path:str, fold_idx:int):
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
                box = ellipse2box(major_r, minor_r, angle, cx, cy)
                boxes.append(box)
        if len(boxes) > 0:
            boxes = np.array(boxes)
            targets.append(boxes)

    return ids, targets


class FDDBDataset(Dataset):
    """FDDB torch.utils.data.Dataset Instance"""

    __phases__ = ("train", "val")
    __folds__ = tuple((i+1 for i in range(10)))
    def __init__(self, source_dir:str, phase:str='train', folds: List[int] = None,
            transform=None, target_transform=None, transforms=None):
        assert phase in FDDBDataset.__phases__,f"given phase {phase} is not valid, must be one of: {self.__phases__}"
        super().__init__()

        # TODO handle train or val stage
        folds = list(self.__folds__) if folds is None else folds

        self.ids = []
        self.targets = []
        self.source_dir = source_dir
        for fold_idx in folds:
            assert fold_idx in self.__folds__, f"given fold {fold_idx} is not in the fold list"
            ids,targets = load_single_annotation_fold(source_dir, fold_idx)
            self.ids += ids
            self.targets += targets

        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    def __getitem__(self, idx:int):
        img = self._load_image(self.ids[idx])
        target_boxes = self.targets[idx].copy()

        if self.transforms:
            img,target_boxes = self.transforms(img,target_boxes)

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target_boxes = self.target_transform(target_boxes)

        return img,target_boxes

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _load_image(img_file_path:str):
        """loads rgb image using given file path

        Args:
            img_path (str): image file path to load

        Returns:
            np.ndarray: rgb image as np.ndarray
        """
        img = imageio.imread(img_file_path)
        if not img.flags['C_CONTIGUOUS']:
            # if img is not contiguous than fix it
            img = np.ascontiguousarray(img, dtype=img.dtype)

        if len(img.shape) == 4:
            # found RGBA
            img = img[:,:,:3]

        if len(img.shape) == 2:
            # gray image found
            img = np.stack([img,img,img], axis=-1)
        return img