import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import math
from .anchor import Anchor

class Matcher():
    def __init__(self, rf_start_offsets:List[int]=None, rf_strides:List[int]=None,
        rf_sizes:List[int]=None, scales:List[Tuple[int,int]]=None, input_size:int=640, **kwargs):
        """[summary]

        Args:
            rf_start_offsets (List[int]): [description]
            rf_strides (List[int]): [description]
            rf_sizes (List[int]): [description]
            scales (List[Tuple[int,int]]): [description]
            input_size (int): input spatial size (assumed rectangle)
        """
        fmaps:List[int] = [input_size // rf_size - 1 for rf_size in rf_strides]

        # TODO check if all list lenghts are matched

        self.heads:List = []

        for rf_stride, rf_start_offset, rf_size, (lower_scale, upper_scale),fmap in zip(rf_strides,
                rf_start_offsets, rf_sizes, scales, fmaps):
            self.heads.append(
                {
                    "anchor" : Anchor(rf_stride, rf_start_offset, rf_size),
                    "sl_range": (int(math.floor(lower_scale * 0.9)), lower_scale),
                    "su_range": (upper_scale, int(math.ceil(upper_scale * 1.1))),
                    "fmap": (fmap,fmap)
                }
            )

    def __call__(self, gt_boxes:torch.Tensor) -> Dict:
        """Generates target cls and regs with masks, using ground truth boxes

        Args:
            gt_boxes (torch.Tensor): N',4 as xmin,ymin,xmax,ymax

        Returns:
            Dict:
                heads: List[Dict[str,torch.Tensor]]
                    (0):
                        target_cls          : fh',fw'   | torch.float
                        ignore_cls_mask     : fh',fw'   | torch.bool
                        target_regs         : fh',fw',4 | torch.float
                        reg_mask            : fh',fw'   | torch.bool
                    ...
                gt_boxes: N',4
        """
        heads:List = []

        for head in self.heads:
            fh,fw = head['fmap']
            sl_range = head['sl_range']
            su_range = head['su_range']
            anchor_generator = head['anchor']
            anchors = anchor_generator(fh,fw)

            target_cls = torch.zeros(fh,fw)                         # 0: bg  | 1: match
            ignore_cls_mask = torch.zeros(fh,fw, dtype=torch.bool)  # False: accept | True: ignore
            target_regs = torch.zeros(fh,fw,4)                      # tx1,ty1,tx2,ty2
            reg_mask = torch.zeros(fh,fw, dtype=torch.bool)         # 0: ignore | True: accept

            rf_normalizer = anchor_generator.rf_size/2
            rf_centers = (anchors[:,:, [0,1]] + anchors[:,:, [2,3]]) / 2

            # rf_anchors: fh x fw x 4 as xmin,ymin,xmax,ymax
            if gt_boxes.size(0) == 0:
                heads.append({
                    'target_cls':target_cls,
                    'ignore_cls_mask':ignore_cls_mask,
                    'target_regs':target_regs,
                    'reg_mask':reg_mask
                })
                continue

            # select max face dim as `face scale` (defined in the paper)
            face_scales,_ = (gt_boxes[:,[2,3]] - gt_boxes[:,[0,1]]).max(dim=1)

            # only select gt boxes that falls between scales
            gt_select_cond = (face_scales >= sl_range[0]) & (face_scales <= su_range[1])

            s_gt_boxes = gt_boxes[gt_select_cond, :4]
            face_scales = face_scales[gt_select_cond]

            if s_gt_boxes.size(0) == 0:
                heads.append({
                    'target_cls':target_cls,
                    'ignore_cls_mask':ignore_cls_mask,
                    'target_regs':target_regs,
                    'reg_mask':reg_mask
                })
                continue

            # lookup ignores
            lower_ignore_cond = (face_scales >= sl_range[0]) & (face_scales <= sl_range[1])
            upper_ignore_cond = (face_scales >= su_range[0]) & (face_scales <= su_range[1])
            gt_ignore_box_ids, = torch.where(lower_ignore_cond | upper_ignore_cond)

            # lets match
            for box_idx,(x1,y1,x2,y2) in enumerate(s_gt_boxes):
                cond_x = (rf_centers[:,:,0] > x1) & (rf_centers[:,:,0] < x2)
                cond_y = (rf_centers[:,:,1] > y1) & (rf_centers[:,:,1] < y2)
                match = cond_x & cond_y
                # match: fh,fw boolean

                # if there is no match than continue
                if match.sum() == 0:
                    continue

                # if falls in gray scale ignore
                if box_idx in gt_ignore_box_ids:
                    ignore_cls_mask[match] = True
                    continue

                # set as matched
                target_cls[match] += 1

                # set reg targets
                target_regs[match, [0]] = (rf_centers[match, [0]] - x1) / rf_normalizer
                target_regs[match, [1]] = (rf_centers[match, [1]] - y1) / rf_normalizer
                target_regs[match, [2]] = (rf_centers[match, [0]] - x2) / rf_normalizer
                target_regs[match, [3]] = (rf_centers[match, [1]] - y2) / rf_normalizer

            multi_matches = target_cls > 1
            positive_matches = target_cls == 1

            ignore_cls_mask[ multi_matches ] = True
            target_cls[multi_matches] = 1
            reg_mask[positive_matches] = True

            heads.append({
                'target_cls':target_cls,
                'ignore_cls_mask':ignore_cls_mask,
                'target_regs':target_regs,
                'reg_mask':reg_mask
            })

        return {'heads':heads, 'gt_boxes':gt_boxes}

    @staticmethod
    def collate_fn(data):
        # TODO think about train / val / test
        imgs,targets = zip(*data)
        batch = torch.stack(imgs, dim=0)
        """
        for target in targets:
            # target
            {
                "heads": [
                    {
                        "target_cls": torch.Tensor,
                        "ignore_cls_mask": torch.Tensor,
                        "target_regs": torch.Tensor,
                        "reg_mask": torch.Tensor
                    }
                ],
                "gt_boxes": torch.Tensor
            }
        """
        num_of_heads = len(targets[0]['heads'])
        ntargets:List = [
            {
                'target_cls': [],
                'ignore_cls_mask': [],
                'target_regs': [],
                'reg_mask': []
            } for _ in range(num_of_heads)]

        n_gt_boxes:List = []

        for target in targets:
            heads = target['heads']
            n_gt_boxes.append(target['gt_boxes'])
            for i in range(num_of_heads):
                ntargets[i]['target_cls'].append(heads[i]['target_cls'])
                ntargets[i]['ignore_cls_mask'].append(heads[i]['ignore_cls_mask'])
                ntargets[i]['target_regs'].append(heads[i]['target_regs'])
                ntargets[i]['reg_mask'].append(heads[i]['reg_mask'])

        for i,target in enumerate(ntargets):
            for k in target:
                ntargets[i][k] = torch.stack(target[k], dim=0)

        return batch, {'heads':ntargets, 'gt_boxes':n_gt_boxes}