import math
from typing import List, Tuple, Union

import torch
from pytorch_lightning.metrics import Metric

from .functional import average_precision
from .utils import generate_prediction_table


class AveragePrecision(Metric):
    """pytorch_lightning.metrics.Metric instance to calculate binary average precision

    Args:
            iou_threshold (Union[List, float]): iou threshold or list of iou thresholds
            area (str): small or medium or large or None

    Returns:
            [type]: [description]
    """

    __areas__ = {
        "small": (0 ** 2, 32 ** 2),
        "medium": (32 ** 2, 96 ** 2),
        "large": (96 ** 2, math.inf),
    }

    def __init__(self, iou_threshold: Union[List, float] = 0.5, area: str = None):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        if area is None:
            area_range = (0, math.inf)
            area_name = ""
        else:
            assert area in self.__areas__, "given area is not defined"
            area_name = ""
            area_range = self.__areas__[area]

        self.area_name = area_name
        self.area_range = area_range
        self.iou_threshold = (
            iou_threshold if isinstance(iou_threshold, List) else [iou_threshold]
        )
        # [N,5 dimensional as xmin,ymin,xmax,ymax,conf]
        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        # [M,4 dimensional as xmin,ymin,xmax,ymax]
        self.add_state("target_boxes", default=[], dist_reduce_fx=None)

    # pylint: disable=method-hidden
    def update(self, preds: List[torch.Tensor], targets: List[torch.Tensor], **kwargs):
        """
        Args:
                preds (List[torch.Tensor]): [N,5 dimensional as xmin,ymin,xmax,ymax,conf]
                targets (List[torch.Tensor]): [M,4 dimensional as xmin,ymin,xmax,ymax]
        """
        # pylint: disable=no-member
        if isinstance(preds, List):
            self.pred_boxes += preds
        else:
            self.pred_boxes.append(preds)

        if isinstance(targets, List):
            self.target_boxes += targets
        else:
            self.target_boxes.append(targets)

    # pylint: disable=method-hidden
    def compute(self):
        """Calculates average precision"""
        # pylint: disable=no-member

        return average_precision(
            self.pred_boxes,
            self.target_boxes,
            iou_thresholds=self.iou_threshold,
            area_range=self.area_range,
        )

    def get_precision_recall_curve(self) -> Tuple[torch.Tensor, torch.Tensor]:
        table = generate_prediction_table(self.preds_boxes, self.targets_boxes)
        # table: List[torch.Tensor] as IoU | Conf | Area | Best | Target Idx
        table = torch.cat(table, dim=0)
        # table: torch.Tensor(N, 5) as IoU | Conf | Area | Best | Target Idx

        if table.size(0) == 0:
            # pylint: disable=not-callable
            return torch.tensor([0], dtype=torch.float32)

        # sort table by confidance scores
        table = table[table[:, 1].argsort(descending=True), :]

        N = table.size(0)
        M = sum([target.size(0) for target in self.targets_boxes])

        # set as fp if lower than iou threshold
        # ! mean value will be used for iou threshold
        iou_threshold = sum(self.iou_threshold) / len(self.iou_threshold)
        table[table[:, 0] < iou_threshold, 3] = 0.0

        accumulated_tp = torch.cumsum(table[:, 3], dim=0)

        precision = accumulated_tp / torch.arange(1, N + 1, dtype=torch.float32)
        recall = accumulated_tp / (M + 1e-16)

        return precision, recall
