from typing import List, Tuple
import torch
import torchvision.ops.boxes as box_ops

def average_precision(predictions: List, targets: List,
        iou_threshold: float = 0.5) -> torch.Tensor:
    """Calculates average precision for given inputs

    Args:
        predictions (List): [Ni,5 dimensional as xmin,ymin,xmax,ymax,conf]
        targets (List): [Mi,4 dimensional as xmin,ymin,xmax,ymax]
        iou_threshold (float, optional): iou threshold for ap score. Defaults to 0.5.

    Raises:
        AssertionError: [description]

    Returns:
        torch.Tensor: average precision score
    """
    assert len(predictions) == len(targets), "prediction and ground truths must be equal in lenght"
    assert len(predictions) > 0, "given input list lenght must be greater than 0"
    device = predictions[0].device

    sorted_table, M = _generate_prediction_table(predictions, targets, device=device)
    N = sorted_table.size(0)

    if N == 0:
        # pylint: disable=not-callable
        return torch.tensor([0], dtype=torch.float32, device=device)

    accumulated_tp = torch.zeros(sorted_table.size(0), dtype=torch.float32, device=device)
    accumulated_fp = torch.zeros(sorted_table.size(0), dtype=torch.float32, device=device)

    sorted_table[sorted_table[:, 0] < iou_threshold, 1] = 0.
    tp = 0
    fp = 0
    for i, row in enumerate(sorted_table):
        # row : 3 as iou,tp,confidence
        if row[1] == 1.:
            tp += 1
        else:
            fp += 1

        accumulated_tp[i] = tp
        accumulated_fp[i] = fp

    precision = accumulated_tp / torch.arange(1, N+1, dtype=torch.float32, device=device)
    recall = accumulated_tp / (M + 1e-16)

    unique_recalls = recall.unique_consecutive()
    auc = torch.empty(unique_recalls.size(0), dtype=torch.float32, device=device)
    # pylint: disable=not-callable
    last_value = torch.tensor(0, dtype=torch.float32, device=device)

    for i, recall_value in enumerate(unique_recalls):
        mask = recall == recall_value # N,
        p_mul = precision[mask].max() # get max p
        auc[i] = p_mul * (recall_value-last_value)
        last_value = recall_value

    return auc.sum()

def _generate_prediction_table(predictions: List, targets: List, device: str = 'cpu') -> Tuple[torch.Tensor, int]:
    """Generates prediction table

    Args:
        predictions (List): [ni,5 as xmin,ymin,xmax,ymax,confidence] total of N prediction (n0 + n1 + n2 ...)
        targets (List): [mi,4 as xmin,ymin,xmax,ymax] total of M ground truths (m0 + m1 + m2 ...)
        device (str): name of the device {cpu | cuda}

    Returns:
        Tuple[torch.Tensor, int]:
            torch.Tensor -- N,3 dimensional matrix as iou,best,confidence
            int -- total number of targets
    """

    table = []
    M = 0
    for pred, gt in zip(predictions, targets):
        mi = gt.size(0)
        ni = pred.size(0)
        if mi == 0:
            if ni != 0:
                tb = torch.zeros(ni, 3, dtype=torch.float32, device=device)
                tb[:, 2] = pred[:, 4]
                table.append(tb)
            continue
        elif ni == 0:
            M += mi
            continue
        M += mi
        ious = box_ops.box_iou(pred[:, :4], gt) # ni,mi vector
        iou_values, iou_indexes = ious.max(dim=1)
        ious = torch.stack([iou_values, iou_indexes.float(), pred[:, 4]]).t() # ni,3
        sort_pick = ious[:, 0].argsort(dim=0, descending=True) # ni,3
        ious = ious[sort_pick].contiguous() # ni,3
        tb = ious.clone() # ni,3
        mask = [True for i in range(gt.size(0))] # mi,
        for i, iou in enumerate(ious):
            index = int(iou[1].long())
            if mask[index]:
                tb[i][1] = 1.   # assign best
                mask[index] = False
            else:
                tb[i][1] = 0.   # assign ignore
        table.append(tb) # ni,3

    if len(table) == 0:
        return torch.empty(0, 3, device=device), M

    # return N,3 tensor as iou_value,best,confidence
    table = torch.cat(table, dim=0)
    select = table[:, 2].argsort(descending=True)

    return table[select].contiguous(), M
