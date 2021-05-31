from typing import List, Dict, Tuple
import torch
import torch.nn as nn

from .blocks import (
    CSPDarknet53Tiny,
    PANetTiny,
    YoloHead
)

from ...utils.box import jaccard_centered

class YOLOv4(nn.Module):

    __CONFIGS__ = {
        "tiny":{
            "input_shape": (-1, 3, 416, 416),
            "strides": [32, 16],
            "anchors": [
                [
                    [177.1328, 261.872 ],
                    [116.896 , 172.8896],
                    [ 85.1136, 126.7136]
                ],
                [
                    [ 64.1888,  95.9712],
                    [ 39.104 ,  58.7392],
                    [ 20.7168,  29.4528]
                ]
            ],
            'head_infeatures': [512, 256],
            'neck_features': 512
        }
    }

    def __init__(self, config: Dict = {}, **kwargs):
        super().__init__()

        assert "input_shape" in config, "`input_shape` must be defined in the config"
        assert "strides" in config, "`strides` must be defined in the config"
        assert "anchors" in config, "`anchors` must be defined in the config"
        assert "head_infeatures" in config, "`head_infeatures` must be defined in the config"
        assert "neck_features" in config, "`neck_features` must be defined in the config"

        # TODO consider another config that is not tiny

        anchors = config['anchors']
        strides = config['strides']
        input_shape = config['input_shape']
        head_infeatures = config['head_infeatures']
        neck_features = config['neck_features']

        self.input_shape = input_shape
        self.backbone = CSPDarknet53Tiny()
        self.neck = PANetTiny(neck_features)
        self.heads = nn.ModuleList([
            YoloHead(in_features, stride, _anchors)
            for stride, _anchors, in_features in zip(strides, anchors, head_infeatures)
        ])

        self.cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.reg_loss_fn = nn.MSELoss(reduction='none')

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """preprocessed image batch
        Args:
            x (torch.Tensor): B x C x H x W
        Returns:
            List[torch.Tensor]: list of logits as B x FH x FW x nA x 5
                (0:4) reg logits
                (4:5) cls logits
        """
        out, residual = self.backbone(x)
        outs = self.neck(out, residual)
        logits: List[torch.Tensor] = []

        for head_idx, head in enumerate(self.heads):
            head_logits = head(outs[head_idx])
            # head_logits: b x nA*(4+1) x fh x fw
            b, nC, fh, fw = head_logits.shape
            logits.append(
                head_logits.permute(0,2,3,1).reshape(b, fh, fw, nC//5, 5))
        return logits

    def logits_to_preds(self, logits: List[torch.Tensor]) -> torch.Tensor:
        """Applies postprocess to given logits

        Args:
            logits (List[torch.Tensor]): list of logits as B x FH x FW x nA x 5
                (0:4) reg logits
                (4:5) cls logits
        Returns:
            torch.Tensor: as preds with shape of B x N x 5 where x1,y1,x2,y2,score
        """
        preds: List[torch.Tensor] = []

        for head_idx, head in enumerate(self.heads):

            scores = torch.sigmoid(
                logits[head_idx][:, :, :, :, [4]]).flatten(start_dim=1, end_dim=3)
            # scores: B x n x 1

            boxes = head.logits_to_boxes(
                logits[head_idx][:, :, :, :, :4]).flatten(start_dim=1, end_dim=3)
            # boxes: B x n x 4

            preds.append(
                # B x n x 5 as x1,y1,xbox2,y2,score
                torch.cat([boxes, scores], dim=2).contiguous()
            )

        # concat channelwise: B x N x 5
        return torch.cat(preds, dim=1).contiguous()

    def compute_loss(self, logits: List[torch.Tensor],
            raw_targets: List[Dict], hparams: Dict = {}) -> Dict[str, torch.Tensor]:
        """Computes loss using given logits and raw targets

        Args:
            logits (List[torch.Tensor]): list of logits as B x FH x FW x nA x 5
                (0:4) reg logits
                (4:5) cls logits
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            hparams (Dict, optional): model hyperparameter dict. Defaults to {}.

        Returns:
            Dict[str, torch.Tensor]: loss values as key value pairs

        """
        batch_size = len(raw_targets)

        nC_shapes = [head_logits.shape[1:4] for head_logits in logits]

        logits = torch.cat(
            [head_logits.view(batch_size, -1, 5) for head_logits in logits],
            dim=1
        )
        # logits: b, n, 5

        reg_logits = logits[:, :, :4]
        # reg_logits: b, n, 4

        cls_logits = logits[:, :, 4]
        # cls_logits: b, n

        targets = self.build_targets(nC_shapes, raw_targets,
            logits.dtype, logits.device)
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
        keep_cls = max(num_of_positives*neg_select_ratio, 100)

        cls_loss = torch.cat([pos_cls_loss, neg_cls_loss[order][: keep_cls]]).mean()

        if pos_mask.sum() > 0:
            reg_loss = self.reg_loss_fn(reg_logits[pos_mask], reg_targets[pos_mask]).mean()
        else:
            reg_loss = torch.tensor(0, dtype=logits.dtype, device=logits.device, requires_grad=True) # pylint: disable=not-callable

        loss = cls_loss + reg_loss

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss
        }

    def build_targets(self, nC_shapes: List[Tuple[int, int, int]], raw_targets: List[Dict],
            dtype=torch.float32, device="cpu") -> torch.Tensor:
        """build model targets using given logits and raw targets

        Args:
            nC_shapes (List[Tuple[int, int]]): nC shapes for each head, [(fh,fw,nA), ...]
            raw_targets (List[Dict]): list of dicts as;
                "target_boxes": torch.Tensor(N, 4)
            dtype : dtype of the tensor. Defaults to torch.float32.
            device (str): device of the tensor. Defaults to 'cpu'.

        Returns:
            torch.Tensor: targets as B x N x 5 where (0:4) reg targets (4:5) objectness targets
        """
        ignore_iou_threshold = 0.5 # make it parametric

        all_anchors = torch.cat([head.anchor.anchors for head in self.heads], dim=0)
        # all_anchors: torch.Tensor(nA*nheads, 2) as w, h

        all_ids = []
        for head_idx, head in enumerate(self.heads):
            for anchor_idx in range(head.anchor.anchors.size(0)):
                all_ids.append((head_idx, anchor_idx))

        bs = len(raw_targets)

        objness_targets = [
            torch.zeros(bs, fh, fw, nA, dtype=dtype, device=device)
            for fh,fw,nA in nC_shapes]

        reg_targets = [
            torch.zeros(bs, fh, fw, nA, 4, dtype=dtype, device=device)
            for fh,fw,nA in nC_shapes]

        # for each image
        for batch_idx, targets in enumerate(raw_targets):
            target_boxes = targets["target_boxes"]
            # target_boxes: torch.Tensor(N, 4)
            if target_boxes.size(0) == 0:
                continue

            gt_wh = target_boxes[:, [2, 3]] - target_boxes[:, [0, 1]]
            gt_centers = (target_boxes[:, [2, 3]] + target_boxes[:, [0, 1]]) / 2

            centered_ious = jaccard_centered(all_anchors, gt_wh)
            # centered_ious: torch.Tensor(nA*nH, N)
            best_anchor_ids = centered_ious.argmax(dim=0)
            # best_anchor_ids: torch.Tensor(N,)

            for n_idx in range(target_boxes.size(0)):
                head_idx, anchor_idx = all_ids[best_anchor_ids[n_idx]]
                grid_x, grid_y = (gt_centers[n_idx] / self.heads[head_idx].anchor.stride).floor().long()
                matches, = torch.where(centered_ious[:, n_idx] > ignore_iou_threshold)

                for matched_anchor_idx in matches:
                    if matched_anchor_idx == best_anchor_ids[n_idx]:
                        continue
                    i_head_idx, i_anchor_idx = all_ids[matched_anchor_idx]

                    i_grid_x, i_grid_y = (gt_centers[n_idx] / self.heads[i_head_idx].anchor.stride).floor().long()

                    objness_targets[i_head_idx][batch_idx, i_grid_y, i_grid_x, i_anchor_idx] = -1

                if objness_targets[head_idx][batch_idx, grid_y, grid_x, anchor_idx] == 1:
                    # double match ignore it
                    objness_targets[head_idx][batch_idx, grid_y, grid_x, anchor_idx] = -1
                    continue

                tx = gt_centers[n_idx][0] / self.heads[head_idx].anchor.stride - grid_x
                ty = gt_centers[n_idx][1] / self.heads[head_idx].anchor.stride - grid_y
                tw = torch.log(1e-16 + (gt_wh[n_idx][0] / all_anchors[best_anchor_ids[n_idx]][0]))
                th = torch.log(1e-16 + (gt_wh[n_idx][1] / all_anchors[best_anchor_ids[n_idx]][1]))

                objness_targets[head_idx][batch_idx, grid_y, grid_x, anchor_idx] = 1
                reg_targets[head_idx][batch_idx, grid_y, grid_x, anchor_idx, :] = torch.stack([tx,ty,tw,th])

        return objness_targets, reg_targets




    def configure_optimizers(self, **hparams):
        # TODO
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, weight_decay=0)

        return optimizer
