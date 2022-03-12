from typing import List, Tuple, Dict
import numpy as np

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

        self._curve = None

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

    def compute(self) -> Dict[str, torch.Tensor]:
        stages = ["easy", "medium", "hard"]
        curve = dict()

        for stage in stages:
            curve[stage] = np.zeros((self.threshold_steps, 2), dtype=np.float32)

        normalized_preds = self.normalize_scores(
            [pred.cpu().numpy().astype(np.float32) for pred in self.preds]
        )

        gt_boxes = [gt_boxes.cpu().numpy().astype(np.float32) for gt_boxes in self.targets]

        ignore_flags = list()
        for labels in self.labels:
            i_flags = dict(
                easy=np.zeros((len(labels),), dtype=np.int32),
                medium=np.zeros((len(labels),), dtype=np.int32),
                hard=np.zeros((len(labels),), dtype=np.int32),
            )

            for i, label in enumerate(labels):
                if label == "face_ignore":
                    i_flags["easy"][i] = 1
                    i_flags["medium"][i] = 1
                    i_flags["hard"][i] = 1
                if label == "face_hard":
                    i_flags["easy"][i] = 1
                    i_flags["medium"][i] = 1
                if label == "face_medium":
                    i_flags["easy"][i] = 1

            ignore_flags.append(i_flags)

        total_faces = dict()
        for stage in stages:
            total_faces[stage] = 0

        for preds, gts, i_flags in zip(normalized_preds, gt_boxes, ignore_flags):
            # skip if no gts
            if gts.shape[0] == 0:
                continue

            # count keeped gts
            for stage in stages:
                total_faces[stage] += (i_flags[stage] == 0).sum()

            if preds.shape[0] == 0:
                continue

            # gts: M, 4 as x1, y1, x2, y2
            # preds: N, 5 as x1, y1, x2, y2, normalized_score

            # sort preds by descending order
            preds = preds[(-preds[:, -1]).argsort(), :]

            # evaluate single image per each stage
            for stage in stages:
                match_counts, ignore_pred_mask = self.evaluate_single_image(
                    preds, gts, i_flags[stage]
                )
                # match_counts: N,
                # ignore_pred_mask: N,

                # calculate image pr
                curve[stage] += self.calculate_image_pr(preds, ignore_pred_mask, match_counts)

        for i in range(self.threshold_steps):
            for stage in stages:
                curve[stage][i, 0] = curve[stage][i, 1] / curve[stage][i, 0]
                curve[stage][i, 1] = curve[stage][i, 1] / total_faces[stage]

        # cache curve
        self._curve = curve

        scores = dict()
        for stage in stages:
            propose = curve[stage][:, 0]
            recall = curve[stage][:, 1]

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
            scores[stage] = torch.tensor(
                ((recall[points + 1] - recall[points]) * propose[points + 1]).sum()
            )

        return scores

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

        # TODO implement numpy version
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

    def reset(self):
        self._curve = None
        return super().reset()