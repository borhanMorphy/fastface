from typing import List,Tuple
import torch
from torchvision.ops import boxes as box_ops
from tqdm import tqdm

def calculate_AP(predictions:List, ground_truths:List, iou_threshold:float=.5) -> torch.Tensor:
    """Calculates average precision
    Arguments:
        predictions {List} -- [Ni,5 dimensional as xmin,ymin,xmax,ymax,conf]
        ground_truths {List} -- [Ni,4 dimensional as xmin,ymin,xmax,ymax]
    Keyword Arguments:
        iou_threshold {float} -- iou threshold to decide true positive (default: {.5})
    Returns:
        torch.Tensor -- [description]
    """
    sorted_table,M = generate_prediction_table(predictions,ground_truths) # N,3 as iou,best,confidence with sorted by confidence
    N = sorted_table.size(0)

    if N == 0:
        return torch.tensor([0], dtype=torch.float32)

    accumulated_tp = torch.zeros(sorted_table.size(0), dtype=torch.float32)
    accumulated_fp = torch.zeros(sorted_table.size(0), dtype=torch.float32)

    sorted_table[sorted_table[:, 0] < iou_threshold, 1] = 0.
    tp = 0
    fp = 0
    for i,row in enumerate(sorted_table):
        # row : 3 as iou,tp,confidence
        if row[1] == 1.:
            tp += 1
        else:
            fp += 1

        accumulated_tp[i] = tp
        accumulated_fp[i] = fp

    precision = accumulated_tp / torch.arange(1,N+1, dtype=torch.float32)
    recall = accumulated_tp / (M + 1e-16)

    unique_recalls = recall.unique_consecutive()
    auc = torch.empty(unique_recalls.size(0), dtype=torch.float32)
    last_value = torch.tensor(0, dtype=torch.float32)

    for i,recall_value in enumerate(unique_recalls):
        mask = recall == recall_value # N,
        p_mul = precision[mask].max() # get max p
        auc[i] = p_mul * (recall_value-last_value)
        last_value = recall_value

    return auc.sum()

def generate_prediction_table(predictions:List, ground_truths:List) -> Tuple[torch.Tensor,int]:
    """Generates prediction table
    Arguments:
        predictions {List} -- [ni,5 as xmin,ymin,xmax,ymax,confidence] total of N prediction (n0 + n1 + n2 ...)
        ground_truths {List} -- [mi,4 as xmin,ymin,xmax,ymax] total of M ground truths (m0 + m1 + m2 ...)
    Returns:
        Tuple
            torch.Tensor -- N,3 dimensional matrix as iou,best,confidence
            M -- total gt count
    """

    table = []
    M = 0
    for pred,gt in zip(predictions,ground_truths):
        mi = gt.size(0)
        ni = pred.size(0)
        if mi == 0:
            if ni != 0:
                tb = torch.zeros(ni,3, dtype=torch.float32)
                tb[:, 2] = pred[:, 4]
                table.append(tb)
            continue
        elif ni == 0:
            M += mi
            continue
        M += mi
        ious = box_ops.box_iou(pred[:,:4],gt) # ni,mi vector
        iou_values,iou_indexes = ious.max(dim=1)
        ious = torch.stack([iou_values,iou_indexes.float(), pred[:, 4]]).t() # ni,3
        sort_pick = ious[:,0].argsort(dim=0,descending=True) # ni,3
        ious = ious[sort_pick].contiguous() # ni,3
        tb = ious.clone() # ni,3
        mask = [True for i in range(gt.size(0))] # mi,
        for i,iou in enumerate(ious):
            index = int(iou[1].long())
            if mask[index]:
                tb[i][1] = 1.   # assign best
                mask[index] = False
            else:
                tb[i][1] = 0.   # assign ignore
        table.append(tb) # ni,3

    if len(table) == 0:
        return torch.empty(0,3),M

    # return N,3 tensor as iou_value,best,confidence
    table = torch.cat(table,dim=0)
    select = table[:, 2].argsort(descending=True)

    return table[select].contiguous(),M

def caclulate_means(total_metrics):
    means = {}
    """
        [
            {'loss':0, 'a_loss':1,}
        ]
    """
    for metrics in total_metrics:
        for k,v in metrics.items():
            if k in means:
                means[k].append(v)
            else:
                means[k] = [v]

    for k,v in means.items():
        means[k] = sum(means[k]) / len(means[k])
    return means

def roi_recalls(predictions:List[torch.Tensor],
        ground_truths:List[torch.Tensor], iou_thresholds:torch.Tensor):
    # preds [torch.tensor(N,5), ...]
    # gts [torch.tensor(N,4), ...]
    th_size = iou_thresholds.size(0)
    total_gts = 0.
    total_hits = torch.zeros(th_size, device=predictions[0].device)

    for preds,gts in zip(predictions,ground_truths):
        total_targets = gts.size(0)
        ious = box_ops.box_iou(preds[:,:4],gts) # N,4 | M,4 => N,M
        hits = torch.zeros((th_size,total_targets), dtype=torch.bool, device=gts.device)
        for i in range(th_size):
            for j in range(total_targets):
                hits[i,j] = ious[:,j].max() > iou_thresholds[i]
            total_hits[i] += hits[i].sum()
        total_gts += total_targets

    return total_hits/total_gts