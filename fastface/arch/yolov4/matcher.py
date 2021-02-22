import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .utils import AnchorGenerator

class Matcher():
    def __init__(self):
        pass # TODO

    def __call__(self, gt_boxes:torch.Tensor) -> Dict:
        """Generates target cls and regs with masks, using ground truth boxes

        Args:
            gt_boxes (torch.Tensor): N',4 as xmin,ymin,xmax,ymax

        Returns:
            Dict:
        """
        pass # TODO

    @staticmethod
    def collate_fn(data):
        pass # TODO