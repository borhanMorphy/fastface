from pytorch_lightning.metrics import Metric
from typing import List,Tuple
import torch
from torchvision.ops import boxes as box_ops

class AveragePrecision(Metric):
    """Calculates average precision"""

    def __init__(self, dist_sync_on_step=False, iou_threshold:float=.5):
        super().__init__(dist_sync_on_step=dist_sync_on_step,
            compute_on_step=False)
        # TODO set iou_threshold as list
        self.iou_threshold = iou_threshold
        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("gt_boxes", default=[], dist_reduce_fx=None)

    def update(self, preds: List[torch.Tensor], targets: List[torch.Tensor]):
        """
        Arguments:
            preds {List} -- [Ni,5 dimensional as xmin,ymin,xmax,ymax,conf]
            targets {List} -- [Ni,4 dimensional as xmin,ymin,xmax,ymax]
        
        """
        self.pred_boxes += preds
        self.gt_boxes += targets

    def compute(self):
        sorted_table,M = self.generate_prediction_table(
            self.pred_boxes,self.gt_boxes) # N,3 as iou,best,confidence with sorted by confidence

        N = sorted_table.size(0)

        if N == 0: return torch.tensor([0], dtype=torch.float32)

        accumulated_tp = torch.zeros(sorted_table.size(0), dtype=torch.float32)
        accumulated_fp = torch.zeros(sorted_table.size(0), dtype=torch.float32)

        sorted_table[sorted_table[:, 0] < self.iou_threshold, 1] = 0.
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

    @staticmethod
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
        dtype = ground_truths[0].dtype
        device = ground_truths[0].device
        
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
            ious = box_ops.box_iou(pred[:,:4].to(device,dtype),gt) # ni,mi vector
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