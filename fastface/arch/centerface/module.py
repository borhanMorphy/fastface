import math
from typing import Dict, List, Tuple

import albumentations as A
import torch
import torch.nn as nn
from cv2 import cv2

from ... import utils
from ..base import ArchInterface
from .blocks import FPN, CenterFaceHead, MobileNetV2
from .config import CenterFaceConfig
from .loss import SoftBinaryFocalLossWithLogits
from .utils import gaussian_radius, get_gaussian_kernel


class CenterFace(nn.Module, ArchInterface):
    def __init__(self, config: CenterFaceConfig):
        super().__init__()
        self.config = config

        if config.backbone == "mobilenetv2":
            backbone = MobileNetV2.from_pretrained()
        else:
            raise AssertionError("undefined backbone")

        self.nn = nn.Sequential(
            backbone,
            FPN(
                config.fpn_in_features,
                config.fpn_out_feature,
                upsample_method=config.fpn_upsample_method,
            ),
            CenterFaceHead(config.fpn_out_feature, config.num_landmarks),
        )

        self.register_buffer(
            "grids",
            utils.box.generate_grids(
                1500 / config.output_stride, 1500 / config.output_stride
            ),
            persistent=False,
        )

        # define losses
        self.cls_loss = SoftBinaryFocalLossWithLogits(
            beta=config.cls_loss_beta,
            alpha=config.cls_loss_alpha,
            gamma=config.cls_loss_gamma,
            reduction="none",
        )
        self.offset_loss = nn.SmoothL1Loss(reduction="none")
        self.wh_loss = nn.SmoothL1Loss(reduction="none")
        self.landmark_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.nn(batch)

    def compute_preds(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute predictions with given logits

        Args:
            logits (torch.Tensor): logits as tensor with shape of (B x FH x FW x C)
                channels: 0:1 (cls) | 1:3 (wh) | 3:5 (offset) | 5: (landmarks)
        Returns:
            torch.Tensor: model predictions as tensor with shape of B x N x (5 + 2*l)
                xmin, ymin, xmax, ymax, score, *landmarks
                where `l` is number of landmarks.
        """
        # TODO apply pooling as soft-NMS

        bs, fh, fw, _ = logits.shape
        heatmap = torch.sigmoid(logits[:, :, :, 0])

        centers = (
            self.grids[:fh, :fw, :].repeat(bs, 1, 1, 1) + 0.5 + logits[:, :, :, 3:5]
        ) * self.config.output_stride
        wh = torch.exp(logits[:, :, :, 1:3]) * self.config.output_stride

        x1y1 = centers - wh / 2
        x2y2 = x1y1 + wh

        landmark_logits = logits[:, :, :, 5:].reshape(
            bs, fh, fw, self.config.num_landmarks, 2
        )
        # bs x fh x fw x num_landmarks x 2
        landmarks = (landmark_logits * wh.unsqueeze(3) + centers.unsqueeze(3)).flatten(
            start_dim=3
        )
        # bs x fh x fw x (num_landmarks * 2)

        preds = torch.cat([x1y1, x2y2, heatmap.unsqueeze(3), landmarks], dim=3)
        # preds: bs x fh x fw x (5 + num_landmarks * 2)

        return preds.flatten(start_dim=1, end_dim=2)

    def compute_loss(
        self,
        logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # TODO pydoc

        true_mask = target_logits[..., 0] == 1
        num_positives = true_mask.sum()
        num_positives = max(num_positives, 1)

        b_ids, fh_ids, fw_ids = torch.where(true_mask)

        cls_loss = (
            self.cls_loss(logits[..., 0], target_logits[..., 0]).sum() / num_positives
        )

        wh_loss = (
            self.wh_loss(
                logits[b_ids, fh_ids, fw_ids, 1:3],
                target_logits[b_ids, fh_ids, fw_ids, 1:3],
            ).sum()
            / num_positives
        )

        offset_loss = (
            self.offset_loss(
                logits[b_ids, fh_ids, fw_ids, 3:5],
                target_logits[b_ids, fh_ids, fw_ids, 3:5],
            ).sum()
            / num_positives
        )

        if self.config.num_landmarks > 0:
            bs, fh, fw, _ = target_logits.shape

            target_l_logits = target_logits[..., 5:].reshape(
                bs, fh, fw, self.config.num_landmarks, 2
            )
            l_logits = logits[..., 5:].reshape(bs, fh, fw, self.config.num_landmarks, 2)

            l_mask = ~torch.all((target_l_logits == 0).flatten(start_dim=3), dim=-1)

            l_b_ids, l_fh_ids, l_fw_ids = torch.where(l_mask)
            l_num_positives = max(l_mask.sum(), 1)

            landmark_loss = (
                self.landmark_loss(
                    l_logits[l_b_ids, l_fh_ids, l_fw_ids],
                    target_l_logits[l_b_ids, l_fh_ids, l_fw_ids],
                ).sum()
                / l_num_positives
            )
        else:
            landmark_loss = torch.zeros(1, requires_grad=True).sum()

        # weighted sum of the loss
        loss = (
            (cls_loss * self.config.loss_lambda_cls)
            + (wh_loss * self.config.loss_lambda_wh)
            + (offset_loss * self.config.loss_lambda_offset)
            + (landmark_loss * self.config.loss_lambda_landmark)
        )

        return {
            "loss": loss,
            "cls_loss": cls_loss.detach(),
            "offset_loss": offset_loss.detach(),
            "wh_loss": wh_loss.detach(),
            "landmark_loss": landmark_loss.detach(),
        }

    def build_targets(
        self, batch: torch.Tensor, raw_targets: List[Dict]
    ) -> torch.Tensor:
        # TODO pydoc
        batch_size, _, h, w = batch.shape

        fh = h // self.config.output_stride
        fw = w // self.config.output_stride

        dtype = batch.dtype
        device = batch.device

        target_logits = torch.zeros(
            *(batch_size, fh, fw, 5 + self.config.num_landmarks * 2),
            dtype=dtype,
            device=device
        )
        # channels: 0:1 (cls) | 1:3 (wh) | 3:5 (offset) | 5: (landmarks)

        for i, targets in enumerate(raw_targets):
            target_boxes = targets["bboxes"].to(device, dtype)
            centers = (target_boxes[:, [0, 1]] + target_boxes[:, [2, 3]]) / 2
            boxes_wh = target_boxes[:, [2, 3]] - target_boxes[:, [0, 1]]
            # applied floor op
            fmap_centers = (centers / self.config.output_stride).long()
            # fmap_centers: N, 2

            # set target cls
            for j, center in enumerate(fmap_centers):
                # center: x, y
                target_w, target_h = boxes_wh[j].cpu()
                target_w = math.ceil(target_w.item())
                target_h = math.ceil(target_h.item())

                # ref: https://github.com/xingyizhou/CenterNet/blob/2b7692c377c6686fb35e473dac2de6105eed62c6/src/lib/utils/image.py#L126
                radius = gaussian_radius((target_h, target_w), min_overlap=0.7)

                diameter = 2 * radius + 1
                sigma = diameter / 6

                heatmap = get_gaussian_kernel(
                    fw,
                    fh,
                    sigma=sigma,
                    center_point=center.cpu().numpy(),
                )

                heatmap = torch.from_numpy(heatmap).to(device, dtype)
                # heatmap: fh x fw
                target_logits[i, :, :, 0] = torch.maximum(
                    heatmap, target_logits[i, :, :, 0]
                )

                # set target landmarks if exists
                if "keypoints" in targets and not targets["keypoint_ids"][j].endswith(
                    "_ignore"
                ):
                    grid_x, grid_y = center

                    keypoints = targets["keypoints"][j].to(device, dtype)
                    # keypoints ni, 2

                    target_logits[i, grid_y, grid_x, 5:] = (
                        (keypoints - centers[j]) / boxes_wh[j]
                    ).flatten()

            # set target reg
            true_indexes_x, true_indexes_y = fmap_centers.t()
            w = target_boxes[:, 2] - target_boxes[:, 0]
            h = target_boxes[:, 3] - target_boxes[:, 1]

            target_logits[i, true_indexes_y, true_indexes_x, 0] = 1

            target_logits[i, true_indexes_y, true_indexes_x, 1] = torch.log(
                (w / self.config.output_stride) + 1e-8
            )
            target_logits[i, true_indexes_y, true_indexes_x, 2] = torch.log(
                (h / self.config.output_stride) + 1e-8
            )

            # set target offset
            target_logits[i, true_indexes_y, true_indexes_x, 3:5] = (
                centers / self.config.output_stride
            ) - fmap_centers

        return target_logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.scheduler_lr_milestones,
            gamma=self.config.scheduler_lr_gamma,
        )
        return [optimizer], [scheduler]

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (
            self.config.input_channel,
            self.config.input_height,
            self.config.input_width,
        )

    @property
    def train_transforms(self):
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(self.config.img_size, self.config.img_size, p=0.5),
                A.ColorJitter(p=0.5),
                A.LongestMaxSize(max_size=self.config.img_size),
                A.PadIfNeeded(
                    min_width=self.config.img_size,
                    min_height=self.config.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=self.config.min_face_area,
                min_visibility=0.7,
            ),
            keypoint_params=A.KeypointParams(
                format="xy", label_fields=["keypoint_ids"], remove_invisible=False
            ),
        )

    @property
    def transforms(self):
        return A.Compose(
            [
                A.LongestMaxSize(max_size=self.config.img_size),
                A.PadIfNeeded(
                    min_width=self.config.img_size,
                    min_height=self.config.img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )
