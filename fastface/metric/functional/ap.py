import math
from typing import List, Tuple

import torch

from ..utils import generate_prediction_table


def average_precision(
    predictions: List,
    targets: List,
    iou_thresholds: List[float] = [0.5],
    area_range: Tuple[int, int] = (0, math.inf),
) -> torch.Tensor:
    """Calculates average precision for given inputs

    Args:
        predictions (List): [N,5 dimensional as xmin,ymin,xmax,ymax,conf]
        targets (List): [M,4 dimensional as xmin,ymin,xmax,ymax]
        iou_thresholds (List[float], optional): list of iou thresholds. Defaults to [0.5].
        area_range (Tuple[int, int], optional): min box area and max box area. Defaults to (0, math.inf).

    Returns:
        torch.Tensor: average precision
    """
    assert len(predictions) == len(
        targets
    ), "prediction and ground truths must be equal in lenght"
    assert len(predictions) > 0, "given input list lenght must be greater than 0"

    table = generate_prediction_table(predictions, targets)
    # table: List[torch.Tensor] as IoU | Conf | Area | Best | Target Idx
    table = torch.cat(table, dim=0)
    # table: torch.Tensor(N, 5) as IoU | Conf | Area | Best | Target Idx

    if table.size(0) == 0:
        # pylint: disable=not-callable
        return torch.tensor([0], dtype=torch.float32)

    # sort table by confidance scores
    table = table[table[:, 1].argsort(descending=True), :]

    # filter by area
    # TODO handle if area is 0
    mask = (table[:, 2] > area_range[0]) & (table[:, 2] < area_range[1])
    table = table[mask, :]

    N = table.size(0)

    if N == 0:
        # pylint: disable=not-callable
        return torch.tensor([0], dtype=torch.float32)

    # TODO make it better
    all_targets = torch.cat(targets, dim=0)
    areas = (all_targets[:, 2] - all_targets[:, 0]) * (
        all_targets[:, 3] - all_targets[:, 1]
    )
    mask = (areas > area_range[0]) & (areas < area_range[1])
    M = areas[mask].size(0)

    aps = []

    # for each iou threshold
    for iou_threshold in iou_thresholds:
        ntable = table.clone()
        # set as fp if lower than iou threshold
        ntable[ntable[:, 0] < iou_threshold, 3] = 0.0

        accumulated_tp = torch.cumsum(ntable[:, 3], dim=0)

        precision = accumulated_tp / torch.arange(1, N + 1, dtype=torch.float32)
        recall = accumulated_tp / (M + 1e-16)

        unique_recalls = recall.unique_consecutive()
        auc = torch.empty(unique_recalls.size(0), dtype=torch.float32)
        # pylint: disable=not-callable
        last_value = torch.tensor(0, dtype=torch.float32)

        for i, recall_value in enumerate(unique_recalls):
            mask = recall == recall_value  # N,
            p_mul = precision[mask].max()  # get max p
            auc[i] = p_mul * (recall_value - last_value)
            last_value = recall_value
        aps.append(auc.sum())
    return sum(aps) / len(aps)
