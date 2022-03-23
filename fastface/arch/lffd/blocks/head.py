import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .anchor import Anchor
from .conv import conv1x1


class LFFDHead(nn.Module):
    def __init__(
        self,
        head_idx: int,
        in_features: int,
        features: int,
        min_face_size: int,
        max_face_size: int,
        rf_stride: int,
    ):
        super().__init__()
        self.head_idx = head_idx
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        self.min_gray_face_scale = math.floor(min_face_size * 0.9)
        self.max_gray_face_scale = math.ceil(max_face_size * 1.1)

        self.anchor = Anchor(rf_stride, rf_stride - 1, max_face_size)

        self.det_conv = nn.Sequential(conv1x1(in_features, features), nn.ReLU())

        self.cls_head = nn.Sequential(
            conv1x1(features, features), nn.ReLU(), conv1x1(features, 1)
        )

        self.reg_head = nn.Sequential(
            conv1x1(features, features), nn.ReLU(), conv1x1(features, 4)
        )

        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.reg_loss_fn = nn.MSELoss(reduction="none")

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.det_conv(features)

        cls_logits = self.cls_head(data)
        # (b,1,h,w)
        reg_logits = self.reg_head(data)
        # (b,c,h,w)
        return reg_logits, cls_logits

    def logits_to_boxes(self, reg_logits: torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,fh,fw,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        _, fh, fw, _ = reg_logits.shape

        rf_centers = self.anchor.rf_centers[:fh, :fw].repeat(1, 1, 2)
        # rf_centers: fh,fw,4 cx1,cy1,cx1,cy1

        return rf_centers - reg_logits * self.anchor.rf_normalizer

    def build_targets(
        self, batch: torch.Tensor, raw_targets: List[Dict]
    ) -> torch.Tensor:
        batch_size, _, height, width = batch.shape
        dtype = batch.dtype
        device = batch.device

        fh, fw = self.anchor.estimate_fmap(height, width)

        rf_centers = self.anchor.rf_centers[:fh, :fw, :]

        target_regs = torch.zeros((batch_size, fh, fw, 4), dtype=dtype, device=device)
        # target_regs: batch_size, fh, fw, 4
        target_cls = torch.zeros((batch_size, fh, fw, 1), dtype=dtype, device=device)

        target_assignment = torch.zeros(
            (batch_size, fh, fw, 1), dtype=dtype, device=device
        )
        # 0: negative
        # 1: positive
        # -1: ignore

        # TODO vectorize the loop
        for batch_idx in range(batch_size):
            bboxes = raw_targets[batch_idx]["bboxes"]
            face_scales, _ = (bboxes[:, [2, 3]] - bboxes[:, [0, 1]]).max(dim=1)

            box_assignments = torch.zeros((bboxes.shape[0]), dtype=dtype, device=device)
            # 0: negative
            # 1: positive
            # -1: ignore

            # find positive boxes
            pos_box_mask = (face_scales > self.min_face_size) & (
                face_scales < self.max_face_size
            )
            box_assignments[pos_box_mask] = 1

            # find ignore boxes
            ignore_box_mask = (~pos_box_mask) & (
                (face_scales >= self.min_gray_face_scale)
                & (face_scales <= self.max_gray_face_scale)
            )
            box_assignments[ignore_box_mask] = -1

            # ground truth assignment
            for box_id, (xmin, ymin, xmax, ymax) in enumerate(bboxes):
                matches = (
                    (rf_centers[..., 0] >= xmin) & (rf_centers[..., 0] <= xmax)
                ) & ((rf_centers[..., 1] >= ymin) & (rf_centers[..., 1] <= ymax))

                matched_fh, matched_fw = torch.where(matches)

                # matches: fh x fw
                # assignment
                target_assignment[
                    batch_idx, matched_fh, matched_fw, :
                ] = box_assignments[box_id]
                # inc cls +1 if matched
                if box_assignments[box_id] == 1:
                    target_cls[batch_idx, matched_fh, matched_fw, 0] += 1
                    target_regs[batch_idx, matched_fh, matched_fw, 0] = (
                        rf_centers[matched_fh, matched_fw, 0] - xmin
                    )
                    target_regs[batch_idx, matched_fh, matched_fw, 1] = (
                        rf_centers[matched_fh, matched_fw, 1] - ymin
                    )
                    target_regs[batch_idx, matched_fh, matched_fw, 2] = (
                        rf_centers[matched_fh, matched_fw, 0] - xmax
                    )
                    target_regs[batch_idx, matched_fh, matched_fw, 3] = (
                        rf_centers[matched_fh, matched_fw, 1] - ymax
                    )
                    target_regs[
                        batch_idx, matched_fh, matched_fw, :
                    ] /= self.anchor.rf_normalizer

            # check for double matched rfs and ignore it
            ignore_fh, ignore_fw = torch.where(target_cls[batch_idx, :, :, 0] > 1)
            target_assignment[batch_idx, ignore_fh, ignore_fw, :] = -1

        return torch.cat([target_regs, target_cls, target_assignment], dim=3)

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
        hard_neg_mining_ratio: int,
    ) -> torch.Tensor:
        pos_mask = target_logits[..., 5] == 1
        neg_mask = target_logits[..., 5] == 0

        pos_cls_loss = self.cls_loss_fn(
            logits[..., 4][pos_mask],
            target_logits[..., 4][pos_mask],
        )
        neg_cls_loss = self.cls_loss_fn(
            logits[..., 4][neg_mask],
            target_logits[..., 4][neg_mask],
        )

        num_of_positives = pos_cls_loss.shape[0]
        num_of_negatives = neg_cls_loss.shape[0]

        order = neg_cls_loss.argsort(descending=True)
        keep_cls = max(
            min(num_of_positives * hard_neg_mining_ratio, num_of_negatives), 100
        )

        cls_loss = torch.cat([pos_cls_loss, neg_cls_loss[order][:keep_cls]])

        if pos_mask.sum() > 0:
            reg_loss = self.reg_loss_fn(
                logits[..., :4][pos_mask], target_logits[..., :4][pos_mask]
            )
        else:
            reg_loss = torch.tensor(
                [[0, 0, 0, 0]],
                dtype=logits.dtype,
                device=logits.device,
                requires_grad=True,
            )  # pylint: disable=not-callable

        return cls_loss.mean(), reg_loss.mean()
