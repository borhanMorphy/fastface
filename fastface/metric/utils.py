from typing import List

import torch

from ..utils.box import jaccard_vectorized


def generate_prediction_table(
    preds: List[torch.Tensor], targets: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Generates prediction table

    Args:
        preds (List[torch.Tensor]): list of predictions as [torch.Tensor(N,5), ...] xmin,ymin,xmax,ymax,score
        targets (List[torch.Tensor]): list of targets as [torch.Tensor(M,4), ...] xmin,ymin,xmax,ymax

    Returns:
        List[torch.Tensor]: list of table as [torch.Tensor(N,4), ...] IoU, Conf, Area, Best, Target Idx
    """
    assert len(preds) == len(targets), "length of predictions and targets must be same"

    table = []
    # IoU | Conf | Area | Best
    for pred, gt in zip(preds, targets):
        N = pred.size(0)
        M = gt.size(0)

        if M == 0:
            single_table = torch.zeros(N, 5)
            single_table[:, 4] = -1
            # IoU | Conf | Area | Best | GT IDX
            single_table[:, 1] = pred[:, 4]
            table.append(single_table)
            continue
        elif N == 0:
            continue

        ious = jaccard_vectorized(pred[:, :4], gt)
        # ious: N x M
        iou_vals, match_ids = torch.max(ious, dim=1)
        # iou_vals: N,
        # match_ids: N,

        best_matches = torch.zeros(N, dtype=torch.long)
        # best_matches: N,

        areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])

        single_table = torch.stack(
            [iou_vals, pred[:, 4], areas[match_ids], best_matches, match_ids.float()],
            dim=1,
        )

        single_table = single_table[single_table[:, 0].argsort(dim=0, descending=True)]
        for gt_idx in range(M):
            (match_ids,) = torch.where(single_table[:, 4] == gt_idx)
            if match_ids.size(0) == 0:
                continue
            # set best
            single_table[match_ids[0], 3] = 1

        table.append(single_table)

    return table
