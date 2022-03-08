from typing import List, Tuple

import torch
import torchvision.ops.boxes as box_ops
from torchmetrics.metric import Metric


class WiderFaceAP(Metric):
    """torchmetrics.metrics.Metric instance to calculate widerface average precision

    Args:
        iou_threshold (float): widerface AP score IoU threshold, default is 0.5 (used value in paper)
    """
    # this implementation heavily inspired by: https://github.com/wondervictor/WiderFace-Evaluation

    is_differentiable = False
    preds: List[torch.Tensor]
    targets: List[torch.Tensor]
    labels: List[List[str]]

    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.iou_threshold = iou_threshold
        self.threshold_steps = 1000
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def update(
        self,
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
		labels: List[List[str]],
    ):
        """_summary_

        Args:
            preds (List[torch.Tensor]): list of N, 5 as xmin, ymin, xmax, ymax, score
            targets (List[torch.Tensor]): list of N, 4 as xmin, ymin, xmax, ymax
            labels (List[List[str]]): list of list labels
        """
        self.preds += preds
        self.targets += targets
        self.labels += labels

    def compute(self) -> torch.Tensor:
        curve = torch.zeros((self.threshold_steps, 2), dtype=torch.float32)

        normalized_preds = self.normalize_scores(
            [pred.cpu().float() for pred in self.preds]
        )

        gt_boxes = [gt_boxes.cpu().float() for gt_boxes in self.targets]

        # TODO convert this into str => tensor
        ignore_flags = list()
        for labels in self.labels:
            flags = torch.zeros((len(labels),), dtype=torch.int32)
            for i, label in enumerate(labels):
                if label == "face_ignore":
                    flags[i] = 1
            ignore_flags.append(flags)

        total_faces = 0

        for preds, gts, i_flags in zip(normalized_preds, gt_boxes, ignore_flags):
            # skip if no gts
            if gts.shape[0] == 0:
                continue

            # count keeped gts
            total_faces += (i_flags == 0).sum()

            if preds.shape[0] == 0:
                continue
            # gts: M,4 as x1,y1,x2,y2
            # preds: N,5 as x1,y1,x2,y2,norm_score

            # sort preds by descending order
            preds = preds[preds[:, -1].argsort(descending=True), :]

            # evaluate single image
            match_counts, ignore_pred_mask = self.evaluate_single_image(
                preds, gts, i_flags
            )
            # match_counts: N,
            # ignore_pred_mask: N,

            # calculate image pr
            curve += self.calculate_image_pr(preds, ignore_pred_mask, match_counts)

        for i in range(self.threshold_steps):
            curve[i, 0] = curve[i, 1] / curve[i, 0]
            curve[i, 1] = curve[i, 1] / total_faces

        propose = curve[:, 0]
        recall = curve[:, 1]

        # add sentinel values at the end
        # [0] + propose + [0]
        propose = torch.cat([torch.tensor([0.0]), propose, torch.tensor([0.0])])

        # [0] + propose + [1]
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])

        # compute the precision envelope
        for i in range(propose.shape[0] - 1, 0, -1):
            propose[i - 1] = max(propose[i - 1], propose[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        (points,) = torch.where(recall[1:] != recall[:-1])

        # and sum (\Delta recall) * prec
        return ((recall[points + 1] - recall[points]) * propose[points + 1]).sum()

    @staticmethod
    def normalize_scores(batch_preds: List[torch.Tensor]) -> List[torch.Tensor]:
        norm_preds = []
        max_score = 0
        min_score = 1
        for preds in batch_preds:
            if preds.shape[0] == 0:
                continue

            min_score = min(preds[:, -1].min(), min_score)
            max_score = max(preds[:, -1].max(), max_score)

        d = max_score - min_score

        for preds in batch_preds:
            n_preds = preds.clone()
            if preds.shape[0] == 0:
                norm_preds.append(n_preds)
                continue
            n_preds[:, -1] = (n_preds[:, -1] - min_score) / d
            norm_preds.append(n_preds)
        return norm_preds

    def evaluate_single_image(
        self, preds: torch.Tensor, gts: torch.Tensor, ignore: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        N = preds.shape[0]
        M = gts.shape[0]

        ious = box_ops.box_iou(preds[:, :4], gts)
        # ious: N,M

        ignore_pred_mask = torch.zeros((N,), dtype=torch.float32)
        gt_match_mask = torch.zeros((M,), dtype=torch.float32)
        match_counts = torch.zeros((N,), dtype=torch.float32)

        for i in range(N):
            max_iou, max_iou_idx = ious[i, :].max(), ious[i, :].argmax()

            if max_iou >= self.iou_threshold:
                if ignore[max_iou_idx] == 1:  # if matched gt is ignored
                    ignore_pred_mask[i] = 1  # set prediction to be ignored later
                    gt_match_mask[max_iou_idx] = -1  # set gt match as ignored
                elif gt_match_mask[max_iou_idx] == 0:  # if matched gt is not ignored
                    gt_match_mask[max_iou_idx] = 1  # set match as positive

            # count each positive match
            match_counts[i] = (gt_match_mask == 1).sum()

        return match_counts, ignore_pred_mask

    def calculate_image_pr(
        self, preds: torch.Tensor, ignore_pred_mask: torch.Tensor, match_counts: torch.Tensor
    ) -> torch.Tensor:

        pr = torch.zeros((self.threshold_steps, 2), dtype=torch.float32)
        thresholds = torch.arange(0, self.threshold_steps, dtype=torch.float32)
        thresholds = 1 - (thresholds + 1) / self.threshold_steps

        for i, threshold in enumerate(thresholds):

            (pos_ids,) = torch.where(preds[:, 4] >= threshold)
            if len(pos_ids) == 0:
                pr[i, 0] = 0
                pr[i, 1] = 0
            else:
                pos_ids = pos_ids[-1]
                (p_index,) = torch.where(ignore_pred_mask[: pos_ids + 1] == 0)
                pr[i, 0] = len(p_index)
                pr[i, 1] = match_counts[pos_ids]
        return pr
