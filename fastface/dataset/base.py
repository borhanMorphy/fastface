__all__ = ["BaseDataset"]

from typing import List, Dict, Tuple
import copy

from torch.utils.data import Dataset
import numpy as np
import imageio

class _IdentitiyTransforms():
    """Dummy tranforms"""
    def __call__(self, img: np.ndarray, targets: Dict) -> Tuple:
        return img, targets

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

        return (img, targets)

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _load_image(img_file_path: str):
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
            # found RGBA, converting to => RGB
            img = img[:, :, :3]
        elif len(img.shape) == 2:
            # found GRAYSCALE, converting to => RGB
            img = np.stack([img, img, img], axis=-1)

        return np.array(img, dtype=np.uint8)
