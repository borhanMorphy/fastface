import copy
import logging
import os
from typing import Dict, List, Tuple

import checksumdir
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..adapter import download_object

logger = logging.getLogger("fastface.dataset")


class _IdentitiyTransforms:
    """Dummy tranforms"""

    def __call__(self, img: np.ndarray, targets: Dict) -> Tuple:
        return img, targets


def default_collate_fn(batch):
    batch, targets = zip(*batch)
    batch = np.stack(batch, axis=0).astype(np.float32)
    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
    for i, target in enumerate(targets):
        for k, v in target.items():
            if isinstance(v, np.ndarray):
                targets[i][k] = torch.from_numpy(v)

    return batch, targets


class BaseDataset(Dataset):
    def __init__(self, ids: List[str], targets: List[Dict], transforms=None, **kwargs):
        super().__init__()
        assert isinstance(ids, list), "given `ids` must be list"
        assert isinstance(targets, list), "given `targets must be list"
        assert len(ids) == len(targets), "lenght of both lists must be equal"

        self.ids = ids
        self.targets = targets
        self.transforms = _IdentitiyTransforms() if transforms is None else transforms

        # set given kwargs to the dataset
        for key, value in kwargs.items():
            if hasattr(self, key):
                # log warning
                continue
            setattr(self, key, value)

    def __getitem__(self, idx: int) -> Tuple:
        img = self._load_image(self.ids[idx])
        targets = copy.deepcopy(self.targets[idx])

        # apply transforms
        img, targets = self.transforms(img, targets)

        # clip boxes
        targets["target_boxes"] = self._clip_boxes(
            targets["target_boxes"], img.shape[:2]
        )

        # discard zero sized boxes
        targets["target_boxes"] = self._discard_zero_size_boxes(targets["target_boxes"])

        return (img, targets)

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        # TODO pydoc
        height, width = shape
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(min=0, max=width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(min=0, max=height - 1)

        return boxes

    @staticmethod
    def _discard_zero_size_boxes(boxes: np.ndarray) -> np.ndarray:
        # TODO pydoc
        scale = (boxes[:, [2, 3]] - boxes[:, [0, 1]]).min(axis=1)
        return boxes[scale > 0]

    @staticmethod
    def _load_image(img_file_path: str):
        """loads rgb image using given file path

        Args:
            img_path (str): image file path to load

        Returns:
            np.ndarray: rgb image as np.ndarray
        """
        img = imageio.imread(img_file_path)
        if not img.flags["C_CONTIGUOUS"]:
            # if img is not contiguous than fix it
            img = np.ascontiguousarray(img, dtype=img.dtype)

        if len(img.shape) == 4:
            # found RGBA, converting to => RGB
            img = img[:, :, :3]
        elif len(img.shape) == 2:
            # found GRAYSCALE, converting to => RGB
            img = np.stack([img, img, img], axis=-1)

        return np.array(img, dtype=np.uint8)

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
        for img, targets in tqdm(
            self, total=len(self), desc="computing normalized target boxes"
        ):
            if targets["target_boxes"].shape[0] == 0:
                continue
            max_size = max(img.shape)
            normalized_boxes.append(targets["target_boxes"] / max_size)

        return np.concatenate(normalized_boxes, axis=0)

    def get_box_scale_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        bins = map(lambda x: 2 ** x, range(10))
        total_boxes = []
        for _, targets in tqdm(self, total=len(self), desc="getting box sizes"):
            if targets["target_boxes"].shape[0] == 0:
                continue
            total_boxes.append(targets["target_boxes"])

        total_boxes = np.concatenate(total_boxes, axis=0)
        areas = (total_boxes[:, 2] - total_boxes[:, 0]) * (
            total_boxes[:, 3] - total_boxes[:, 1]
        )

        return np.histogram(np.sqrt(areas), bins=list(bins))

    def download(self, urls: List, target_dir: str):
        for k, v in urls.items():

            keys = list(v["check"].items())
            checked_keys = []

            for key, md5hash in keys:
                target_sub_dir = os.path.join(target_dir, key)
                if not os.path.exists(target_sub_dir):
                    checked_keys.append(False)
                else:
                    checked_keys.append(
                        checksumdir.dirhash(target_sub_dir, hashfunc="md5") == md5hash
                    )

            if sum(checked_keys) == len(keys):
                logger.debug("found {} at {}".format(k, target_dir))
                continue

            # download
            adapter = v.get("adapter")
            kwargs = v.get("kwargs", {})
            logger.warning(
                "{} not found in the {}, downloading...".format(k, target_dir)
            )
            download_object(adapter, dest_path=target_dir, **kwargs)
