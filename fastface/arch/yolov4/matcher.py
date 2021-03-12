import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from .anchor import Anchor

from ...utils.box import cxcywh2xyxy, xyxy2cxcywh

def box_iou_wh(anchors: torch.Tensor, gt_box: torch.Tensor) -> torch.Tensor:
    # TODO pydoc
    # anchors: N,2 as w h
    # gt_box: 2 as w h
    gt_box = gt_box.repeat(anchors.size(0), 1)
    min_w = torch.min(anchors[:, 0], gt_box[:, 0])
    min_h = torch.min(anchors[:, 1], gt_box[:, 1])
    max_w = torch.max(anchors[:, 0], gt_box[:, 0])
    max_h = torch.max(anchors[:, 1], gt_box[:, 1])

    return (min_w*min_h) / (max_w*max_h + 1e-16)

class Matcher():
    def __init__(self, anchors: List = None, strides: List[int] = None,
            img_size: int = None, iou_match_threshold: float = 0.5, **kwargs):

        self.iou_match_threshold = iou_match_threshold
        self.heads = [
            # pylint: disable=not-callable
            Anchor(_anchors, img_size, stride)
            for _anchors, stride in zip(anchors, strides)
        ]

    def __call__(self, gt_boxes: torch.Tensor) -> Dict:
        """Generates target cls and regs with masks, using ground truth boxes

        Args:
            gt_boxes (torch.Tensor): N',4 as xmin,ymin,xmax,ymax

        Returns:
            Dict:
                heads: List[Dict[str,torch.Tensor]]
                    (0):
                        target_objectness   : nA x grid_y x grid_x       | torch.float
                        ignore_objectness   : nA x grid_y x grid_x       | torch.bool
                        target_regs         : nA x grid_y x grid_x x 4   | torch.float
                        ignore_preds        : nA x grid_y x grid_x       | torch.bool
                    ...
                gt_boxes: N',4
        """

        device = gt_boxes.device
        dtype = gt_boxes.dtype
        num_of_gts = gt_boxes.size(0)
        heads = []

        # TODO think about empty gt boxes

        for head in self.heads:
            nA = head._anchors.size(0)
            anchors = head._anchors * head._stride # original size
            stride = head._stride
            nGy = head._cached_grids[1]
            nGx = head._cached_grids[0]

            target_objectness = torch.zeros(*(nA, nGy, nGx), device=device, dtype=dtype)
            ignore_objectness = torch.zeros(*(nA, nGy, nGx), device=device, dtype=torch.bool)
            target_regs = torch.zeros(*(nA, nGy, nGx, 4), device=device, dtype=dtype)
            ignore_preds = torch.ones(*(nA, nGy, nGx), device=device, dtype=torch.bool)

            if num_of_gts == 0:
                heads.append(
                    {
                        "target_objectness": target_objectness,
                        "ignore_objectness": ignore_objectness,
                        "target_regs": target_regs,
                        "ignore_preds": ignore_preds
                    }
                )
                continue

            gt_boxes[:, :4] = gt_boxes[:, :4].clamp(min=0)
            gt_boxes_cxcywh = xyxy2cxcywh(gt_boxes[:, :4])

            # select best matched anchors
            ious = torch.stack([box_iou_wh(anchors, gt[2:]) for gt in gt_boxes_cxcywh])
            # ious: torch.Tensor(num_of_gts, nA)
            best_ious, best_anchors = ious.max(dim=1)

            # assign to grid
            gt_centers = gt_boxes_cxcywh[:, :2] # get centers
            gt_centers /= stride # down to grid level
            gx, gy = gt_centers.floor().long().t()
            gx = gx.clamp(min=0, max=nGx-1)
            gy = gy.clamp(min=0, max=nGy-1)

            target_objectness[best_anchors, gy, gx] = 1
            ignore_preds[best_anchors, gy, gx] = False
            target_regs[best_anchors, gy, gx, :] = gt_boxes[:, :4]

            for j in range(num_of_gts):
                # ignore if anchor is not matched, but exceeds iou threshold
                ignore_objectness[ious[j] > self.iou_match_threshold, gy[j], gx[j]] = True

            ignore_objectness[best_anchors, gy, gx] = False

            heads.append(
                {
                    "target_objectness": target_objectness,
                    "ignore_objectness": ignore_objectness,
                    "target_regs": target_regs,
                    "ignore_preds": ignore_preds
                }
            )

        return {
            "heads": heads,
            "gt_boxes": gt_boxes
        }

    @staticmethod
    def collate_fn(data):
        # TODO pydoc
        # TODO think about train / val / test
        imgs,targets = zip(*data)
        batch = torch.stack(imgs, dim=0)
        """
        for target in targets:
            # target
            {
                "heads":[
                    {
                        "target_objectness": torch.Tensor,
                        "ignore_objectness": torch.Tensor,
                        "target_regs": torch.Tensor,
                        "ignore_preds": torch.Tensor
                    },
                    ...
                ]
                "gt_boxes": torch.Tensor
            }
        """

        num_of_heads = len(targets[0]['heads'])
        ntargets:List = [
            {
                "target_objectness": [],
                "ignore_objectness": [],
                "target_regs": [],
                "ignore_preds": []
            } for _ in range(num_of_heads)]

        n_gt_boxes:List = []

        for target in targets:
            heads = target['heads']
            n_gt_boxes.append(target['gt_boxes'])
            for i in range(num_of_heads):
                ntargets[i]["target_objectness"].append(heads[i]["target_objectness"])
                ntargets[i]["ignore_objectness"].append(heads[i]["ignore_objectness"])
                ntargets[i]["target_regs"].append(heads[i]["target_regs"])
                ntargets[i]["ignore_preds"].append(heads[i]["ignore_preds"])

        for i,target in enumerate(ntargets):
            for k in target:
                ntargets[i][k] = torch.stack(target[k], dim=0)

        return batch, {'heads':ntargets, 'gt_boxes':n_gt_boxes}