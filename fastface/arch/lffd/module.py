from typing import List, Dict, Union
import math

import torch
import torch.nn as nn

from .blocks import (
    LFFDBackboneV1,
    LFFDBackboneV2,
    LFFDHead
)

from ...loss import BinaryFocalLoss
from ...utils import box as box_ops

class LFFD(nn.Module):

    __CONFIGS__ = {
        'original':{
            'input_shape': (-1, 3, 640, 640),
            'backbone_name': 'lffd-v1',
            'head_infeatures': [64, 64, 64, 64, 128, 128, 128, 128],
            'head_outfeatures': [128, 128, 128, 128, 128, 128, 128, 128],
            'rf_sizes': [15, 20, 40, 70, 110, 250, 400, 560],
            'rf_start_offsets': [3, 3, 7, 7, 15, 31, 31, 31],
            'rf_strides': [4, 4, 8, 8, 16, 32, 32, 32],
            'scales': [
                (10, 15), (15, 20), (20, 40), (40, 70),
                (70, 110), (110, 250), (250, 400), (400, 560)
            ]
        },

        'slim':{
            'input_shape': (-1, 3, 480, 480),
            'backbone_name': 'lffd-v2',
            'head_infeatures': [64, 64, 64, 128, 128],
            'head_outfeatures': [128, 128, 128, 128, 128],
            'rf_sizes': [20, 40, 80, 160, 320],
            'rf_start_offsets': [3, 7, 15, 31, 63],
            'rf_strides': [4, 8, 16, 32, 64],
            'scales': [
                (10, 20), (20, 40), (40, 80), (80, 160), (160, 320)
            ]
        }
    }

    def __init__(self, config: Union[str, Dict], **kwargs):
        super().__init__()

        if isinstance(config, str):
            assert config in self.__CONFIGS__, "given config {} is invalid".format(config)
            config = self.__CONFIGS__[config].copy()

        assert "input_shape" in config, "`input_shape` must be defined in the config"
        assert "backbone_name" in config, "`backbone_name` must be defined in the config"
        assert "head_infeatures" in config, "`head_infeatures` must be defined in the config"
        assert "head_outfeatures" in config, "`head_outfeatures` must be defined in the config"
        assert "rf_sizes" in config, "`rf_sizes` must be defined in the config"
        assert "rf_start_offsets" in config, "`rf_start_offsets` must be defined in the config"
        assert "rf_strides" in config, "`rf_strides` must be defined in the config"
        assert "scales" in config, "`scales` must be defined in the config"

        backbone_name = config.get('backbone_name')
        head_infeatures = config.get('head_infeatures')
        head_outfeatures = config.get('head_outfeatures')
        rf_sizes = config.get('rf_sizes')
        rf_start_offsets = config.get('rf_start_offsets')
        rf_strides = config.get('rf_strides')

        self.face_scales = config["scales"]

        self.input_shape = config.get('input_shape')

        # TODO check if list lenghts are matched
        if backbone_name == "lffd-v1":
            self.backbone = LFFDBackboneV1(3)
        elif backbone_name == "lffd-v2":
            self.backbone = LFFDBackboneV2(3)
        else:
            raise ValueError(f"given backbone name: {backbone_name} is not valid")

        self.heads = nn.ModuleList([
            LFFDHead(idx+1, infeatures, outfeatures, rf_size, rf_start_offset, rf_stride,
                num_classes=1)
            for idx, (infeatures, outfeatures, rf_size, rf_start_offset, rf_stride) in enumerate(zip(
                head_infeatures, head_outfeatures,rf_sizes ,rf_start_offsets ,rf_strides ))
        ])

        self.cls_loss_fn = BinaryFocalLoss(gamma=2, alpha=1)
        self.reg_loss_fn = nn.MSELoss(reduction='none')

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """preprocessed image batch

        Args:
            batch (torch.Tensor): B x C x H x W

        Returns:
            torch.Tensor: logits as B x 5 x N
                (0:4) reg logits
                (4:5) cls logits
        """
        features = self.backbone(batch)

        logits: List[torch.Tensor] = []
        batch_size = batch.size(0)

        for i, head in enumerate(self.heads):
            reg_logits, cls_logits = head(features[i])
            nreg, fh, fw = reg_logits.shape[1:]

            nc = cls_logits.size(1)
 
            logits.append(
                # concat channel wise
                torch.cat([
                    reg_logits.view(batch_size, nreg, fh*fw),
                    cls_logits.view(batch_size, nc, fh*fw)], dim=1)
            )

        # concat N wise
        return torch.cat(logits, dim=2)

    def postprocess(self, logits: torch.Tensor, input_shape: torch.Size = None):
        """Applies postprocess to given logits

        Args:
            logits (torch.Tensor): as B x 5 x N
                (0:4) reg logits
                (4:5) cls logits
            input_shape (torch.Size, optional): shape of the input. Defaults to None.
        """
        batch_size, _, image_h, image_w = input_shape

        reg_logits = logits[:, :4, :].permute(0, 2, 1)
        # reg_logits: B x N x 4
        cls_logits = torch.sigmoid(logits[:, [4], :].permute(0, 2, 1))
        # cls_logits: B x N x 1

        counter = 0

        preds = []

        for head_idx in range(len(self.heads)):
            fh = image_h // self.heads[head_idx].anchor.rf_stride - 1
            fw = image_w // self.heads[head_idx].anchor.rf_stride - 1
            start_index = counter
            end_index = start_index + fh*fw
            counter += (fh*fw)

            head_reg_logits = reg_logits[:, start_index:end_index, :].view(-1, fh, fw, 4)
            
            boxes = self.heads[head_idx].anchor.logits_to_boxes(head_reg_logits).view(-1, fh*fw, 4)
            # boxes: bs,(fh*fw),4

            scores = cls_logits[:, start_index:end_index, :].view(-1, fh*fw, 1)
            # scores: bs,(fh*fw),1

            preds.append(
                torch.cat([boxes, scores], dim=2) # bs,(fh*fw),5
            )
        preds = torch.cat(preds, dim=1) # bs, N, 5

        # filter with det_threshold
        pick_b, pick_n = torch.where(preds[:, :, 4] >= 0.2)

        boxes = preds[pick_b, pick_n, :4]
        scores = preds[pick_b, pick_n, 4]

        # keep only k
        order = scores.argsort(descending=True)
        boxes = boxes[order][:100, :]
        scores = scores[order][:100]
        batch_ids = pick_b[order][:100]

        # filter with nms
        preds, batch_ids = box_ops.batched_nms(boxes, scores, batch_ids)

        return torch.cat([preds, batch_ids.unsqueeze(1)], dim=1)


    def compute_loss(self, logits: torch.Tensor,
            raw_targets: List[Dict], input_shape: torch.Size = None,
            hparams: Dict = {}) -> Dict[str, torch.Tensor]:
        """Computes loss using given logits and raw targets

        Args:
            logits (torch.Tensor): torch.Tensor(B, 5, N) where;
                (0:4) reg logits
                (4:5) cls logits
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            hparams (Dict, optional): model hyperparameter dict. Defaults to {}.
            input_shape (torch.Size): shape of the input. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: loss values as key value pairs

        """
        # TODO use hparams

        reg_logits = logits[:, :4, :].permute(0, 2, 1)
        # reg_logits: b, 4, n => b, n, 4

        cls_logits = logits[:, [4], :].permute(0, 2, 1)
        # cls_logits: b, 1, n => b, n, 1

        targets = self.build_targets(logits, raw_targets, input_shape=input_shape)
        # targets: b, 5, n

        reg_targets = targets[:, :4, :].permute(0, 2, 1)
        # reg_targets: b, 4, n => b, n, 4

        cls_targets = targets[:, [4], :].permute(0, 2, 1)
        # cls_targets: b, 1, n => b, n, 1

        pos_mask = (cls_targets == 1).squeeze(-1)
        cls_mask = cls_targets != -1

        cls_loss = self.cls_loss_fn(cls_logits[cls_mask], cls_targets[cls_mask]).mean()

        if pos_mask.sum() > 0:
            reg_loss = self.reg_loss_fn(reg_logits[pos_mask], reg_targets[pos_mask]).mean()
        else:
            reg_loss = torch.tensor(0, dtype=logits.dtype, device=logits.device, requires_grad=True)

        loss = cls_loss + reg_loss

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss
        }

    def build_targets(self, logits: torch.Tensor,
            raw_targets: List[Dict], input_shape: torch.Size = None) -> torch.Tensor:
        """build model targets using given logits and raw targets

        Args:
            logits (torch.Tensor): torch.Tensor(B, 5, N)
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            input_shape (torch.Size): shape of the input. Defaults to None.

        Returns:
            torch.Tensor: TODO
        """
        batch_size, _, image_h, image_w = input_shape

        device = logits.device
        dtype = logits.dtype

        fmap_size = logits.size(2)
        batch_target_boxes = []
        batch_target_face_scales = []
        for target in raw_targets:
            t_boxes = target["target_boxes"]
            batch_target_boxes.append(t_boxes)
            
            # select max face dim as `face scale` (defined in the paper)
            batch_target_face_scales.append(
                (t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]).max(dim=1)[0]
            )

        targets = []

        for head_idx, head in enumerate(self.heads):
            min_face_scale, max_face_scale = self.face_scales[head_idx]
            min_gray_face_scale = math.floor(min_face_scale * 0.9)
            max_gray_face_scale = math.ceil(max_face_scale * 1.1)

            fh = image_h // head.anchor.rf_stride - 1
            fw = image_w // head.anchor.rf_stride - 1

            rfs = head.anchor.forward(fh, fw).to(device)
            # rfs: fh x fw x 4 as xmin, ymin, xmax, ymax

            # calculate rf normalizer for the head
            rf_normalizer = head.anchor.rf_size / 2

            # get rf centers
            rf_centers = (rfs[..., [2, 3]] + rfs[..., [0, 1]]) / 2
            # rf_centers: fh x fw x 2 as center_x, center_y

            rfs = rfs.repeat(batch_size, 1, 1, 1)
            # rfs fh x fw x 4 => bs x fh x fw x 4

            head_cls_targets = torch.zeros(*(batch_size, 1, fh, fw), dtype=dtype, device=device) # 0: bg, 1: fg, -1: ignore
            head_reg_targets = torch.zeros(*(batch_size, 4, fh, fw), dtype=dtype, device=device)

            for batch_idx, (target_boxes, target_face_scales) in enumerate(zip(batch_target_boxes, batch_target_face_scales)):
                # for each image in the batch
                if target_boxes.size(0) == 0:
                    continue

                # selected accepted boxes
                head_accept_box_ids, = torch.where((target_face_scales > min_face_scale) & (target_face_scales < max_face_scale))

                # find ignore boxes
                head_ignore_box_ids, = torch.where(((target_face_scales >= min_gray_face_scale) & (target_face_scales <= min_face_scale))\
                    | ((target_face_scales <= max_gray_face_scale) & (target_face_scales >= max_face_scale)))

                for gt_idx, (x1, y1, x2, y2) in enumerate(target_boxes):

                    match_mask = ((x1 < rf_centers[:, :, 0]) & (x2 > rf_centers[:, :, 0])) \
                        & ((y1 < rf_centers[:, :, 1]) & (y2 > rf_centers[:, :, 1]))

                    # match_mask: fh, fw
                    if match_mask.sum() <= 0:
                        continue

                    if gt_idx in head_ignore_box_ids:
                        # if gt is in gray scale, all matches sets as ignore
                        match_fh, match_fw = torch.where(match_mask)

                        # set matches as fg
                        head_cls_targets[batch_idx, [0], match_fh, match_fw] = -1
                        continue
                    elif gt_idx not in head_accept_box_ids:
                        # if gt not in gray scale and not in accepted ids, than skip it
                        continue

                    match_fh, match_fw = torch.where(match_mask & (head_cls_targets[batch_idx, 0, :, :] != -1))
                    double_match_fh, double_match_fw = torch.where((head_cls_targets[batch_idx, 0, :, :] == 1) & match_mask)

                    # set matches as fg
                    head_cls_targets[batch_idx, [0], match_fh, match_fw] = 1

                    head_reg_targets[batch_idx, 0, match_fh, match_fw] = (rf_centers[match_fh, match_fw, 0] - x1) / rf_normalizer
                    head_reg_targets[batch_idx, 1, match_fh, match_fw] = (rf_centers[match_fh, match_fw, 1] - y1) / rf_normalizer
                    head_reg_targets[batch_idx, 2, match_fh, match_fw] = (rf_centers[match_fh, match_fw, 0] - x2) / rf_normalizer
                    head_reg_targets[batch_idx, 3, match_fh, match_fw] = (rf_centers[match_fh, match_fw, 1] - y2) / rf_normalizer

                    # set multi-matches as ignore
                    head_cls_targets[batch_idx, [0], double_match_fh, double_match_fw] = -1

            targets.append(
                torch.cat([
                    head_reg_targets.view(batch_size, 4, fh*fw),
                    head_cls_targets.view(batch_size, 1, fh*fw),
                ], dim=1)
            )
        targets = torch.cat(targets, dim=-1)

        return targets

    def configure_optimizers(self, hparams: Dict ={}):
        # TODO handle here
        return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9, weight_decay=0)