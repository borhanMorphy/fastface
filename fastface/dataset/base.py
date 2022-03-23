import copy
import logging
import os
from typing import Dict, List

import checksumdir
import numpy as np
import torch
from cv2 import cv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..adapter import download_object
from ..utils import discover_versions
from ..utils.cache import get_data_cache_dir

logger = logging.getLogger("fastface.dataset")


def default_collate_fn(batch):
    images = list()
    targets = list()
    for data in batch:
        target = dict()
        for key, val in data.items():
            if key == "image":
                images.append(val)
            else:
                if key not in target:
                    target[key] = list()
                target[key] += val
        if "bboxes" in target:
            target["bboxes"] = np.array(target["bboxes"], dtype=np.float32).reshape(
                -1, 4
            )
        if "keypoints" in target:
            target["keypoints"] = np.array(
                target["keypoints"], dtype=np.float32
            ).reshape(-1, 5, 2)
        targets.append(target)

    batch = np.stack(images, axis=0).astype(np.float32)
    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()

    for i, target in enumerate(targets):
        for k, v in target.items():
            if isinstance(v, np.ndarray):
                targets[i][k] = torch.from_numpy(v)

    return batch, targets


class BaseDataset(Dataset):
    __DATASET_NAME__ = "base"
    __URLS__ = dict()

    def __init__(
        self,
        ids: List[str],
        targets: List[Dict],
        transforms=None,
        drop_keys: List[str] = None,
        **kwargs
    ):
        super().__init__()
        assert isinstance(ids, list), "given `ids` must be list"
        assert isinstance(targets, list), "given `targets must be list"
        assert len(ids) == len(targets), "length of both lists must be equal"

        self.ids = ids
        self.targets = targets
        self.transforms = transforms
        self._drop_keys = drop_keys or list()

        # set given kwargs to the dataset
        for key, value in kwargs.items():
            if hasattr(self, key):
                # log warning
                continue
            setattr(self, key, value)

    def __getitem__(self, idx: int) -> Dict:
        img = self._load_image(self.ids[idx])
        targets = copy.deepcopy(self.targets[idx])

        for drop_key in self._drop_keys:
            if drop_key in targets:
                targets.pop(drop_key)

        # clip boxes
        max_h, max_w = img.shape[:2]
        for i in range(len(targets["bboxes"])):
            x1, y1, x2, y2 = targets["bboxes"][i]
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, max_w)
            y2 = min(y2, max_h)
            targets["bboxes"][i] = [x1, y1, x2, y2]

        if self.transforms:
            # apply transforms
            data = self.transforms(image=img, **targets)
        else:
            data = dict(image=img, **targets)

        return data

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _load_image(img_file_path: str) -> np.ndarray:
        """loads rgb image using given file path

        Args:
            img_path (str): image file path to load

        Returns:
            np.ndarray: rgb image as np.ndarray
        """
        return cv2.imread(img_file_path)[..., [2, 1, 0]]

    def get_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn=default_collate_fn,
        pin_memory: bool = False,
        **kwargs
    ):

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **kwargs
        )

    def get_mean_std(self) -> Dict:
        # TODO pydoc
        mean_sum, mean_sq_sum = np.zeros(3), np.zeros(3)
        for img, _ in tqdm(
            self, total=len(self), desc="calculating mean and std for the dataset"
        ):
            d = img.astype(np.float32) / 255

            mean_sum[0] += np.mean(d[:, :, 0])
            mean_sum[1] += np.mean(d[:, :, 1])
            mean_sum[2] += np.mean(d[:, :, 2])

            mean_sq_sum[0] += np.mean(d[:, :, 0] ** 2)
            mean_sq_sum[1] += np.mean(d[:, :, 1] ** 2)
            mean_sq_sum[2] += np.mean(d[:, :, 2] ** 2)

        mean = mean_sum / len(self)
        std = (mean_sq_sum / len(self) - mean ** 2) ** 0.5

        return {"mean": mean.tolist(), "std": std.tolist()}

    def get_normalized_boxes(self) -> np.ndarray:
        # TODO pydoc
        normalized_boxes = []
        for data in tqdm(
            self, total=len(self), desc="computing normalized target boxes"
        ):
            if len(data["bboxes"]) == 0:
                continue

            max_size = max(data["image"].shape)
            for xmin, ymin, xmax, ymax in data["bboxes"]:
                normalized_boxes.append(
                    [
                        xmin / max_size,
                        ymin / max_size,
                        xmax / max_size,
                        ymax / max_size,
                    ]
                )

        return np.array(normalized_boxes, dtype=np.float32)

    def get_boxes(self) -> np.ndarray:
        total_boxes = []
        for data in tqdm(self, total=len(self), desc="getting box sizes"):
            if len(data["bboxes"]) == 0:
                continue
            total_boxes += data["bboxes"]

        return np.array(total_boxes, dtype=np.float32)

    def _find_missing_requirements(self, source_dir: str) -> List[str]:
        missing_reqs: List[str] = list()

        for req_name in self.__URLS__.keys():
            if not self.__check_requirement(source_dir, req_name):
                missing_reqs.append(req_name)

        return missing_reqs

    def __check_requirement(self, source_dir: str, req_name: str) -> bool:
        for key, expected_md5hash in self.__URLS__[req_name]["check"].items():
            source_sub_dir = os.path.join(source_dir, key)
            if not os.path.exists(source_sub_dir):
                return False

            md5hash = checksumdir.dirhash(source_sub_dir, hashfunc="md5")
            if md5hash != expected_md5hash:
                return False

        return True

    def download(self, target_dir: str = None) -> str:
        if target_dir is None:
            # find target directory
            for version in discover_versions(include_current_version=True):
                target_dir = get_data_cache_dir(
                    suffix=self.__DATASET_NAME__, version=version
                )

                missing_reqs = self._find_missing_requirements(target_dir)
                if len(missing_reqs) == 0:
                    break
            if len(missing_reqs) > 0:
                # if missing requirements exist, then use current version
                target_dir = get_data_cache_dir(suffix=self.__DATASET_NAME__)
        else:
            # use given directory
            missing_reqs = self._find_missing_requirements(target_dir)

        for requirement_name in missing_reqs:
            # download
            adapter = self.__URLS__[requirement_name].get("adapter")
            kwargs = self.__URLS__[requirement_name].get("kwargs", {})
            logger.warning(
                "{} not found in the {}, downloading...".format(
                    requirement_name, target_dir
                )
            )
            download_object(adapter, dest_path=target_dir, **kwargs)

        return target_dir
