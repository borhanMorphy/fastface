__all__ = ["UFDDDataset"]

import os
import numpy as np

from .base import BaseDataset

class UFDDDataset(BaseDataset):
    """UFDD fastface.dataset.BaseDataset Instance"""

    __phases__ = ("train", "val")
    def __init__(self, source_dir: str, phase: str='train',
            transforms=None, **kwargs):

        ids = []
        targets = []

        super().__init__(ids, targets, transforms=transforms, **kwargs)