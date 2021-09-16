from typing import List, Tuple

import numpy as np
import torch
import torchvision.ops.boxes as box_ops
from pytorch_lightning.metrics import Metric


class WiderFaceAP(Metric):
    """pytorch_lightning.metrics.Metric instance to calculate widerface average precision

    Args:
            iou_threshold (float): widerface AP score IoU threshold, default is 0.5

    """

    # this implementation heavily inspired by: https://github.com/wondervictor/WiderFace-Evaluation

    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.iou_threshold = iou_threshold
        self.threshold_steps = 1000
        self.add_state("pred_boxes", default=[], dist_reduce_fx=None)
        self.add_state("gt_boxes", default=[], dist_reduce_fx=None)
        self.add_state("ignore_flags", default=[], dist_reduce_fx=None)

    def update(
        self,
        preds: List[torch.Tensor],
        targets: List[torch.Tensor],
        ignore_flags: List[torch.Tensor] = None,
        **kwargs
    ):
        """
        Arguments:
                preds [List]: [Ni,5 dimensional as xmin,ymin,xmax,ymax,conf]
                targets [List]: [Ni,5 dimensional as xmin,ymin,xmax,ymax]
                ignore_flags [List]: [Ni, dimensional]
        """
        # pylint: disable=no-member
        if isinstance(preds, List):
            self.pred_boxes += preds
        else:
            self.pred_boxes.append(preds)

        if isinstance(ignore_flags, List):
            self.ignore_flags += ignore_flags
        else:
            self.ignore_flags.append(ignore_flags)

        if isinstance(targets, List):
            self.gt_boxes += targets
        else:
            self.gt_boxes.append(targets)

    def compute(self) -> float:
        # pylint: disable=no-member
        curve = np.zeros((self.threshold_steps, 2), dtype=np.float32)

        normalized_preds = self.normalize_scores(
            [pred.float().cpu().numpy() for pred in self.pred_boxes]
        )

        gt_boxes = [gt_boxes.cpu().float().numpy() for gt_boxes in self.gt_boxes]

        ignore_flags = [ignore_flag.cpu().numpy() for ignore_flag in self.ignore_flags]

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

            # sort preds
            preds = preds[(-preds[:, -1]).argsort(), :]

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
        propose = np.concatenate([[0.0], propose, [0.0]])

        # [0] + propose + [1]
        recall = np.concatenate([[0.0], recall, [1.0]])

        # compute the precision envelope
        for i in range(propose.shape[0] - 1, 0, -1):
            propose[i - 1] = max(propose[i - 1], propose[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        (points,) = np.where(recall[1:] != recall[:-1])

        # and sum (\Delta recall) * prec
        ap = ((recall[points + 1] - recall[points]) * propose[points + 1]).sum()

        return ap

    @staticmethod
    def normalize_scores(batch_preds: List[np.ndarray]) -> List[np.ndarray]:
        """[summary]

        Args:
                preds (List[np.ndarray]): [description]

        Returns:
                List[np.ndarray]: [description]
        """
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
            n_preds = preds.copy()
            if preds.shape[0] == 0:
                norm_preds.append(n_preds)
                continue
            n_preds[:, -1] = (n_preds[:, -1] - min_score) / d
            norm_preds.append(n_preds)
        return norm_preds

    def evaluate_single_image(
        self, preds: np.ndarray, gts: np.ndarray, ignore: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        N = preds.shape[0]
        M = gts.shape[0]

        ious = box_ops.box_iou(
            # pylint: disable=not-callable
            torch.tensor(preds[:, :4], dtype=torch.float32),
            # pylint: disable=not-callable
            torch.tensor(gts, dtype=torch.float32),
        ).numpy()
        # ious: N,M

        ignore_pred_mask = np.zeros((N,), dtype=np.float32)
        gt_match_mask = np.zeros((M,), dtype=np.float32)
        match_counts = np.zeros((N,), dtype=np.float32)

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
        self, preds: np.ndarray, ignore_pred_mask: np.ndarray, match_counts: np.ndarray
    ) -> np.ndarray:

        pr = np.zeros((self.threshold_steps, 2), dtype=np.float32)
        thresholds = np.arange(0, self.threshold_steps, dtype=np.float32)
        thresholds = 1 - (thresholds + 1) / self.threshold_steps

        for i, threshold in enumerate(thresholds):

            (pos_ids,) = np.where(preds[:, 4] >= threshold)
            if len(pos_ids) == 0:
                pr[i, 0] = 0
                pr[i, 1] = 0
            else:
                pos_ids = pos_ids[-1]
                (p_index,) = np.where(ignore_pred_mask[: pos_ids + 1] == 0)
                pr[i, 0] = len(p_index)
                pr[i, 1] = match_counts[pos_ids]
        return pr
