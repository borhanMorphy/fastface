from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import albumentations as A
from cv2 import cv2

from .blocks import LFFDBackboneV1, LFFDBackboneV2, LFFDHead
from .config import LFFDConfig
from .transform import RandomScaleSample
from ..base import ArchInterface


class LFFD(nn.Module, ArchInterface):
    # TODO: fixme

    def __init__(self, config: LFFDConfig):
        super().__init__()
        self.config = config

        # TODO check if list lengths are matched
        if config.backbone == "lffd-v1":
            self.backbone = LFFDBackboneV1(config.input_channel)
        elif config.backbone == "lffd-v2":
            self.backbone = LFFDBackboneV2(config.input_channel)
        else:
            raise ValueError("given backbone name: {} is not valid".format(config.backbone))

        min_face_sizes = [config.min_face_size] + list(config.rf_sizes[:-1])
        max_face_sizes = list(config.rf_sizes)

        self.heads = nn.ModuleList(
            [
                LFFDHead(
                    idx + 1,
                    in_features,
                    out_features,
                    min_face_size,
                    max_face_size,
                    rf_stride,
                )
                for idx, (
                    in_features,
                    out_features,
                    min_face_size,
                    max_face_size,
                    rf_stride,
                ) in enumerate(
                    zip(
                        config.head_in_features,
                        config.head_out_features,
                        min_face_sizes,
                        max_face_sizes,
                        config.rf_strides,
                    )
                )
            ]
        )

        self.loss_fn = {
            "cls": nn.BCEWithLogitsLoss(reduction="none"),
            "reg": nn.MSELoss(reduction="none")
        }

        self.initialize_weights()

    def initialize_weights(self):
        def conv_xavier_init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(conv_xavier_init)

    def forward(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Compute logits with given normalized batch of images

        Args:
            batch (torch.Tensor): normalized bach of images with shape of B x C x H x W

        Returns:
            List[torch.Tensor]: list of logits as B x FH x FW x 5
                (0:4) reg logits
                (4:5) cls logits
        """
        features = self.backbone(batch)

        logits: List[torch.Tensor] = []

        for head_idx, head in enumerate(self.heads):
            reg_logits, cls_logits = head(features[head_idx])
            # reg_logits : B x 4 x fh x fw
            # cls_logits : B x 1 x fh x fw
            reg_logits = reg_logits.permute(0, 2, 3, 1)
            cls_logits = cls_logits.permute(0, 2, 3, 1)

            logits.append(
                # concat channel wise
                # B x fh x fw x 4 (+) B x fh x fw x 1
                torch.cat([reg_logits, cls_logits], dim=3)
            )

        return logits

    def compute_preds(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """Compute predictions with given logits

        Args:
            logits (List[torch.Tensor]): list of logits as B x FH x FW x 5
                (0:4) reg logits
                (4:5) cls logits

        Returns:
            torch.Tensor: model predictions as tensor with shape of B x N x (5 + 2*l)
                (0:5) xmin, ymin, xmax, ymax, score, *landmarks
                (5:) where `l` is number of landmarks.
        """
        preds: List[torch.Tensor] = []

        for head_idx, head in enumerate(self.heads):

            scores = torch.sigmoid(logits[head_idx][:, :, :, [4]])
            boxes = head.logits_to_boxes(logits[head_idx][:, :, :, :4])

            preds.append(
                # B x n x 5 as x1,y1,x2,y2,score
                torch.cat([boxes, scores], dim=3)
                .flatten(start_dim=1, end_dim=2)
                .contiguous()
            )

        # concat channelwise: B x N x 5
        return torch.cat(preds, dim=1).contiguous()

    def compute_loss(
        self, logits: List[torch.Tensor], target_logits: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Computes loss using given logits (`forward` output) and raw_targets (`build_targets` output)

        Args:
            logits (List[torch.Tensor]): list of torch.Tensor(B, fh, fw, 5) where;
                (0:4) reg logits
                (4:5) cls logits
            target_logits (List[torch.Tensor]): list of torch.Tensor(B, fh, fw, 6) where;
                (0:4) target reg logits
                (4:5) target cls logits
                (5:6) assigment information, 0: negative, 1: positive, -1: ignore

        Returns:
            Dict[str, torch.Tensor]: loss values as key value pairs

        """

        pos_cls_loss = list()
        neg_cls_loss = list()
        reg_loss = list()
        for head_idx in range(len(self.heads)):
            h_pos_cls_loss, h_neg_cls_loss, h_reg_loss = self.heads[head_idx].compute_loss(
                logits[head_idx], target_logits[head_idx]
            )

            pos_cls_loss.append(h_pos_cls_loss)
            neg_cls_loss.append(h_neg_cls_loss)
            reg_loss.append(h_reg_loss)

        pos_cls_loss = torch.cat(pos_cls_loss, dim=0)
        neg_cls_loss = torch.cat(neg_cls_loss, dim=0)
        reg_loss = torch.cat(reg_loss, dim=0)

        num_of_positives = pos_cls_loss.shape[0]
        num_of_negatives = neg_cls_loss.shape[0]
        
        order = neg_cls_loss.argsort(descending=True)
        keep_cls = max(min(num_of_positives * self.config.hard_neg_mining_ratio, num_of_negatives), 1)

        cls_loss = torch.cat([pos_cls_loss, neg_cls_loss[order][:keep_cls]]).mean()
        reg_loss = reg_loss.mean()

        loss = cls_loss + reg_loss

        return {"loss": loss, "cls_loss": cls_loss, "reg_loss": reg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=self.config.scheduler_milestones,
            gamma=self.config.scheduler_gamma,
            verbose=True,
        )

        return [optimizer], [lr_scheduler]

    def build_targets(self, batch: torch.Tensor, raw_targets: List[Dict]) -> List[torch.Tensor]:
        # include input images in here
        head_target_logits = list()

        for head in self.heads:
            head_target_logits.append(
                head.build_targets(batch, raw_targets)
            )

        return head_target_logits

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (
            self.config.input_channel,
            self.config.input_height,
            self.config.input_width,
        )

    @property
    def train_transforms(self):
        sizes = [self.config.min_face_size] + list(self.config.rf_sizes)
        scales = list(zip(sizes[:-1], sizes[1:]))
        ref_size = max(self.config.input_width, self.config.input_height)
        min_area = (self.config.min_face_size - 1)**2
        return A.Compose(
            [
                # TODO parameterize by config
                A.ColorJitter(
                    brightness=(0.5, 1.5),
                    saturation=(0.5, 1.5),
                    contrast=(0.5, 1.5),
                    hue=0,
                    p=0.5
                ),
                RandomScaleSample(
                    scales,
                    ref_size=ref_size
                ),
                A.PadIfNeeded(min_width=ref_size, min_height=ref_size, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc', label_fields=['labels'], min_visibility=0.3, min_area=min_area
            )
        )

    @property
    def transforms(self):
        ref_size = max(self.config.input_width, self.config.input_height)
        min_area = (self.config.min_face_size - 1)**2
        return A.Compose(
            [
                A.LongestMaxSize(max_size=ref_size),
                A.PadIfNeeded(min_width=ref_size, min_height=ref_size, border_mode=cv2.BORDER_CONSTANT, value=0),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc', label_fields=['labels'], min_visibility=0.3, min_area=min_area
            )
        )
