import torch
from typing import List, Dict

from .module import YOLOv4

from ...utils.box import xyxy2cxcywh

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
    def __init__(self, config: str, iou_ignore_threshold: float = 0.4, **kwargs):

        self.iou_ignore_threshold = iou_ignore_threshold
        self.heads = YOLOv4.get_anchor_generators(config)

    def __call__(self, img: torch.Tensor, gt_boxes: torch.Tensor) -> Dict:
        """Generates target cls and regs with masks, using ground truth boxes

        Args:
            imgs: (torch.Tensor): c x h x w
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
        heads = []

        img_h = img.size(1)
        img_w = img.size(2)

        # TODO think about empty gt boxes

        anchor_sizes = torch.cat([head.anchor_sizes * head.stride for head in self.heads], dim=0)
        grid_size_x = [int(img_w / head.stride) for head in self.heads]
        grid_size_y = [int(img_h / head.stride) for head in self.heads]

        # nA*heads x 2

        target_objectness = [
            torch.zeros(*(head.num_anchors, img_h // head.stride, img_w // head.stride),
                device=device, dtype=dtype)
            for head in self.heads]

        ignore_objectness = [
            torch.zeros(*(head.num_anchors, img_h // head.stride, img_w // head.stride),
                device=device, dtype=torch.bool)
            for head in self.heads]

        target_regs = [
            torch.zeros(*(head.num_anchors, img_h // head.stride, img_w // head.stride), 4,
                device=device, dtype=dtype)
            for head in self.heads]

        for x1, y1, x2, y2 in gt_boxes[:, :4]:

            has_match = [False] * len(self.heads)
            # TODO fix nomember
            gt_wh = torch.tensor([x2-x1,y2-y1], dtype=dtype, device=device)
            gt_center = torch.tensor([(x1+x2)/2, (y1+y2)/2], dtype=dtype, device=device)

            ious = box_iou_wh(anchor_sizes, gt_wh)
            # ious: N,
            for anchor_idx in ious.argsort(descending=True):
                head_idx = anchor_idx.item() // self.heads[0].num_anchors

                head_anchor_idx = anchor_idx % self.heads[head_idx].num_anchors

                grid_x = int((gt_center[0]/img_w) * grid_size_x[head_idx])
                grid_y = int((gt_center[1]/img_h) * grid_size_y[head_idx])

                try:
                    matched = target_objectness[head_idx][head_anchor_idx, grid_y, grid_x]
                except:
                    print(target_objectness[head_idx].shape,grid_y,grid_x,img_h,img_w)
                    print(x1,y1,x2,y2)
                    exit(0)

                if (not matched) and (not has_match[head_idx]):
                    target_objectness[head_idx][head_anchor_idx, grid_y, grid_x] = 1
                    grid_gt_w = (gt_wh[0] / img_w) *  grid_size_x[head_idx]
                    grid_gt_h = (gt_wh[1] / img_h) *  grid_size_y[head_idx]
                    grid_anchor_sizes = anchor_sizes / self.heads[head_idx].stride

                    target_regs[head_idx][head_anchor_idx, grid_y, grid_x, 0] = (gt_center[0]/img_w) * grid_size_x[head_idx] - grid_x
                    target_regs[head_idx][head_anchor_idx, grid_y, grid_x, 1] = (gt_center[1]/img_h) * grid_size_y[head_idx] - grid_y
                    target_regs[head_idx][head_anchor_idx, grid_y, grid_x, 2] = torch.log( (grid_gt_w / grid_anchor_sizes[head_anchor_idx, 0]) + 1e-16 )
                    target_regs[head_idx][head_anchor_idx, grid_y, grid_x, 3] = torch.log( (grid_gt_h / grid_anchor_sizes[head_anchor_idx, 1]) + 1e-16 )

                    has_match[head_idx] = True
                elif (not matched) and (ious[anchor_idx] > self.iou_ignore_threshold):

                    ignore_objectness[head_idx][head_anchor_idx, grid_y, grid_x] = True

        heads = []
        for head_idx in range(len(self.heads)):
            heads.append({
                "target_objectness" : target_objectness[head_idx],
                "target_regs" : target_regs[head_idx],
                "ignore_objectness" : ignore_objectness[head_idx]
            })
        
        return img, {
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
                        "target_regs": torch.Tensor
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
                "target_regs": []
            } for _ in range(num_of_heads)]

        n_gt_boxes:List = []

        for target in targets:
            heads = target['heads']
            n_gt_boxes.append(target['gt_boxes'])
            for i in range(num_of_heads):
                ntargets[i]["target_objectness"].append(heads[i]["target_objectness"])
                ntargets[i]["ignore_objectness"].append(heads[i]["ignore_objectness"])
                ntargets[i]["target_regs"].append(heads[i]["target_regs"])

        for i,target in enumerate(ntargets):
            for k in target:
                ntargets[i][k] = torch.stack(target[k], dim=0)

        return batch, {'heads':ntargets, 'gt_boxes':n_gt_boxes}