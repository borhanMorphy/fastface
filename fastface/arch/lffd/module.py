import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .blocks import LFFDBackboneV1, LFFDBackboneV2, LFFDHead


class LFFD(nn.Module):

    __CONFIGS__ = {
        "original": {
            "input_shape": (-1, 3, 640, 640),
            "backbone_name": "lffd-v1",
            "head_infeatures": [64, 64, 64, 64, 128, 128, 128, 128],
            "head_outfeatures": [128, 128, 128, 128, 128, 128, 128, 128],
            "rf_sizes": [15, 20, 40, 70, 110, 250, 400, 560],
            "rf_start_offsets": [3, 3, 7, 7, 15, 31, 31, 31],
            "rf_strides": [4, 4, 8, 8, 16, 32, 32, 32],
            "scales": [
                (10, 15),
                (15, 20),
                (20, 40),
                (40, 70),
                (70, 110),
                (110, 250),
                (250, 400),
                (400, 560),
            ],
        },
        "slim": {
            "input_shape": (-1, 3, 480, 480),
            "backbone_name": "lffd-v2",
            "head_infeatures": [64, 64, 64, 128, 128],
            "head_outfeatures": [128, 128, 128, 128, 128],
            "rf_sizes": [20, 40, 80, 160, 320],
            "rf_start_offsets": [3, 7, 15, 31, 63],
            "rf_strides": [4, 8, 16, 32, 64],
            "scales": [(10, 20), (20, 40), (40, 80), (80, 160), (160, 320)],
        },
    }

    def __init__(self, config: Dict, **kwargs):
        super().__init__()

        assert "input_shape" in config, "`input_shape` must be defined in the config"
        assert (
            "backbone_name" in config
        ), "`backbone_name` must be defined in the config"
        assert (
            "head_infeatures" in config
        ), "`head_infeatures` must be defined in the config"
        assert (
            "head_outfeatures" in config
        ), "`head_outfeatures` must be defined in the config"
        assert "rf_sizes" in config, "`rf_sizes` must be defined in the config"
        assert (
            "rf_start_offsets" in config
        ), "`rf_start_offsets` must be defined in the config"
        assert "rf_strides" in config, "`rf_strides` must be defined in the config"
        assert "scales" in config, "`scales` must be defined in the config"

        backbone_name = config.get("backbone_name")
        head_infeatures = config.get("head_infeatures")
        head_outfeatures = config.get("head_outfeatures")
        rf_sizes = config.get("rf_sizes")
        rf_start_offsets = config.get("rf_start_offsets")
        rf_strides = config.get("rf_strides")

        self.face_scales = config["scales"]

        self.input_shape = config.get("input_shape")

        # TODO check if list lenghts are matched
        if backbone_name == "lffd-v1":
            self.backbone = LFFDBackboneV1(3)
        elif backbone_name == "lffd-v2":
            self.backbone = LFFDBackboneV2(3)
        else:
            raise ValueError(f"given backbone name: {backbone_name} is not valid")

        self.heads = nn.ModuleList(
            [
                LFFDHead(
                    idx + 1,
                    infeatures,
                    outfeatures,
                    rf_size,
                    rf_start_offset,
                    rf_stride,
                    num_classes=1,
                )
                for idx, (
                    infeatures,
                    outfeatures,
                    rf_size,
                    rf_start_offset,
                    rf_stride,
                ) in enumerate(
                    zip(
                        head_infeatures,
                        head_outfeatures,
                        rf_sizes,
                        rf_start_offsets,
                        rf_strides,
                    )
                )
            ]
        )

        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.reg_loss_fn = nn.MSELoss(reduction="none")

    def forward(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """preprocessed image batch

        Args:
            batch (torch.Tensor): B x C x H x W

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

    def logits_to_preds(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """Applies postprocess to given logits

        Args:
            logits (List[torch.Tensor]): list of logits as B x FH x FW x 5
                (0:4) reg logits
                (4:5) cls logits
        Returns:
            torch.Tensor: as preds with shape of B x N x 5 where x1,y1,x2,y2,score
        """
        preds: List[torch.Tensor] = []

        for head_idx, head in enumerate(self.heads):
            batch_size, fh, fw, _ = logits[head_idx].shape

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
        self, logits: List[torch.Tensor], raw_targets: List[Dict], hparams: Dict = {}
    ) -> Dict[str, torch.Tensor]:
        """Computes loss using given logits and raw targets

        Args:
            logits (List[torch.Tensor]): list of torch.Tensor(B, fh, fw, 5) where;
                (0:4) reg logits
                (4:5) cls logits
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            hparams (Dict, optional): model hyperparameter dict. Defaults to {}.

        Returns:
            Dict[str, torch.Tensor]: loss values as key value pairs

        """
        batch_size = len(raw_targets)
        neg_select_ratio = hparams.get("ratio", 10)

        fmap_shapes = [head_logits.shape[1:3] for head_logits in logits]

        logits = torch.cat(
            [head_logits.view(batch_size, -1, 5) for head_logits in logits], dim=1
        )
        # logits: b, n, 5

        reg_logits = logits[:, :, :4]
        # reg_logits: b, n, 4

        cls_logits = logits[:, :, 4]
        # cls_logits: b, n

        targets = self.build_targets(
            fmap_shapes, raw_targets, logits.dtype, logits.device
        )
        # targets: b, n, 5

        reg_targets = targets[:, :, :4]
        # reg_targets: b, n, 4

        cls_targets = targets[:, :, 4]
        # cls_targets: b, n

        pos_mask = cls_targets == 1
        neg_mask = cls_targets == 0
        num_of_positives = pos_mask.sum()

        pos_cls_loss = self.cls_loss_fn(cls_logits[pos_mask], cls_targets[pos_mask])
        neg_cls_loss = self.cls_loss_fn(cls_logits[neg_mask], cls_targets[neg_mask])
        order = neg_cls_loss.argsort(descending=True)
        keep_cls = max(num_of_positives * neg_select_ratio, 100)

        cls_loss = torch.cat([pos_cls_loss, neg_cls_loss[order][:keep_cls]]).mean()

        if pos_mask.sum() > 0:
            reg_loss = self.reg_loss_fn(
                reg_logits[pos_mask], reg_targets[pos_mask]
            ).mean()
        else:
            reg_loss = torch.tensor(
                0, dtype=logits.dtype, device=logits.device, requires_grad=True
            )  # pylint: disable=not-callable

        loss = cls_loss + reg_loss

        return {"loss": loss, "cls_loss": cls_loss, "reg_loss": reg_loss}

    def build_targets(
        self,
        fmap_shapes: List[Tuple[int, int]],
        raw_targets: List[Dict],
        dtype=torch.float32,
        device="cpu",
    ) -> torch.Tensor:
        """build model targets using given logits and raw targets

        Args:
            fmap_shapes (List[Tuple[int, int]]): feature map shapes for each head, [(fh,fw), ...]
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            dtype : dtype of the tensor. Defaults to torch.float32.
            device (str): device of the tensor. Defaults to 'cpu'.

        Returns:
            torch.Tensor: targets as B x N x 5 where (0:4) reg targets (4:5) cls targets
        """

        batch_target_boxes = []
        batch_target_face_scales = []

        batch_size = len(raw_targets)

        for target in raw_targets:
            t_boxes = target["target_boxes"]
            batch_target_boxes.append(t_boxes)

            # select max face dim as `face scale` (defined in the paper)
            batch_target_face_scales.append(
                0
                if t_boxes.size(0) == 0
                else (t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]).max(dim=1)[0]
            )

        targets = []

        for head_idx, (head, (fh, fw)) in enumerate(zip(self.heads, fmap_shapes)):
            min_face_scale, max_face_scale = self.face_scales[head_idx]
            min_gray_face_scale = math.floor(min_face_scale * 0.9)
            max_gray_face_scale = math.ceil(max_face_scale * 1.1)

            rfs = head.anchor.forward(fh, fw).to(device)
            # rfs: fh x fw x 4 as xmin, ymin, xmax, ymax

            # calculate rf normalizer for the head
            rf_normalizer = head.anchor.rf_size / 2

            # get rf centers
            rf_centers = (rfs[..., [2, 3]] + rfs[..., [0, 1]]) / 2

            # rf_centers: fh x fw x 2 as center_x, center_y

            rfs = rfs.repeat(batch_size, 1, 1, 1)
            # rfs fh x fw x 4 => bs x fh x fw x 4

            head_cls_targets = torch.zeros(
                *(batch_size, fh, fw), dtype=dtype, device=device
            )  # 0: bg, 1: fg, -1: ignore
            head_reg_targets = torch.zeros(
                *(batch_size, fh, fw, 4), dtype=dtype, device=device
            )

            for batch_idx, (target_boxes, target_face_scales) in enumerate(
                zip(batch_target_boxes, batch_target_face_scales)
            ):
                # for each image in the batch
                if target_boxes.size(0) == 0:
                    continue

                # selected accepted boxes
                (head_accept_box_ids,) = torch.where(
                    (target_face_scales > min_face_scale)
                    & (target_face_scales < max_face_scale)
                )

                # find ignore boxes
                (head_ignore_box_ids,) = torch.where(
                    (
                        (target_face_scales >= min_gray_face_scale)
                        & (target_face_scales <= min_face_scale)
                    )
                    | (
                        (target_face_scales <= max_gray_face_scale)
                        & (target_face_scales >= max_face_scale)
                    )
                )

                for gt_idx, (x1, y1, x2, y2) in enumerate(target_boxes):

                    match_mask = (
                        (x1 < rf_centers[:, :, 0]) & (x2 > rf_centers[:, :, 0])
                    ) & ((y1 < rf_centers[:, :, 1]) & (y2 > rf_centers[:, :, 1]))

                    # match_mask: fh, fw
                    if match_mask.sum() <= 0:
                        continue

                    if gt_idx in head_ignore_box_ids:
                        # if gt is in gray scale, all matches sets as ignore
                        match_fh, match_fw = torch.where(match_mask)

                        # set matches as fg
                        head_cls_targets[batch_idx, match_fh, match_fw] = -1
                        continue
                    elif gt_idx not in head_accept_box_ids:
                        # if gt not in gray scale and not in accepted ids, than skip it
                        continue

                    match_fh, match_fw = torch.where(
                        match_mask & (head_cls_targets[batch_idx, :, :] != -1)
                    )
                    double_match_fh, double_match_fw = torch.where(
                        (head_cls_targets[batch_idx, :, :] == 1) & match_mask
                    )

                    # set matches as fg
                    head_cls_targets[batch_idx, match_fh, match_fw] = 1

                    head_reg_targets[batch_idx, match_fh, match_fw, 0] = (
                        rf_centers[match_fh, match_fw, 0] - x1
                    ) / rf_normalizer
                    head_reg_targets[batch_idx, match_fh, match_fw, 1] = (
                        rf_centers[match_fh, match_fw, 1] - y1
                    ) / rf_normalizer
                    head_reg_targets[batch_idx, match_fh, match_fw, 2] = (
                        rf_centers[match_fh, match_fw, 0] - x2
                    ) / rf_normalizer
                    head_reg_targets[batch_idx, match_fh, match_fw, 3] = (
                        rf_centers[match_fh, match_fw, 1] - y2
                    ) / rf_normalizer

                    # set multi-matches as ignore
                    head_cls_targets[batch_idx, double_match_fh, double_match_fw] = -1

            targets.append(
                torch.cat(
                    [
                        head_reg_targets.view(batch_size, -1, 4),
                        head_cls_targets.view(batch_size, -1, 1),
                    ],
                    dim=2,
                )
            )

        # concat n wise
        return torch.cat(targets, dim=1)

    def configure_optimizers(self, hparams: Dict = {}):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=hparams.get("learning_rate", 1e-1),
            momentum=hparams.get("momentum", 0.9),
            weight_decay=hparams.get("weight_decay", 1e-5),
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=hparams.get("milestones", [600000, 1000000, 1200000, 1400000]),
            gamma=hparams.get("gamma", 0.1),
        )

        return [optimizer], [lr_scheduler]
