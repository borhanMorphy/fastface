import torch
from typing import Tuple
import math

# TODO implement RF/Anchor and Ground Truth Matcher

class LFFDMatcher():
    def __init__(self, min_scale:int, max_scale:int):
        # TODO handle multi-class later
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.gsl_range,self.gsu_range = LFFDMatcher.get_gray_scale_range(min_scale,max_scale)

    @staticmethod
    def get_gray_scale_range(sl:int, su:int) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        sl_range = (int(math.floor(sl * 0.9)), sl)
        su_range = (su, int(math.ceil(su * 1.1)))
        return sl_range,su_range

    def __call__(self, rf_anchors:torch.Tensor, gt_boxes:torch.Tensor, device:str='cpu',
            dtype:torch.dtype=torch.float32) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """Matches given receptive field centers and ground truth boxes

        Args:
            rf_anchors (torch.Tensor): fh,fw,4 as x1,y1,x2,y2
            gt_boxes (torch.Tensor): N,4 as x1,y1,x2,y2
            device (str): {cpu:cuda}
            dtype (torch.dtype): torch.float32

        Returns:
            Tuple:
                torch.Tensor(bool): cls_mask {0 | 1} (fh x fw) , (0: negative) (1: positive)
                torch.Tensor(dtype): reg_targets (fh x fw x 4) (target regressions)
                torch.Tensor(bool): ignore_mask {0 | 1} (fh x fw) , (0: not ignored) (1: ignored)
        """
        fh,fw = rf_anchors.shape[:2]

        cls_mask = torch.zeros(*(fh,fw), dtype=torch.bool, device=device)
        reg_targets = torch.zeros(*(fh,fw,4), dtype=dtype, device=device)
        ignore_mask = torch.zeros(*(fh,fw), dtype=torch.bool, device=device)

        # return if gt box is empty
        if gt_boxes.size(0) == 0:
            return cls_mask,reg_targets,ignore_mask
        
        wh = gt_boxes[:,[2,3]] - gt_boxes[:,[0,1]]
        # wh: N,2 as w,h

        # select max face dim as `face scale` (defined in the paper)
        face_scales,_ = wh.max(dim=1)

        # only select gt boxes that falls between scales
        gt_select_cond = (face_scales > self.min_scale) & (face_scales < self.max_scale)
        s_gt_boxes = gt_boxes[gt_select_cond, :]

        # return if no gt box is found in that scale
        if s_gt_boxes.size(0) == 0:
            return cls_mask,reg_targets,ignore_mask

        # lookup ignores
        face_scales = face_scales[gt_select_cond]
        lower_ignore_cond = (face_scales >= self.gsl_range[0]) & (face_scales <= self.gsl_range[1])
        upper_ignore_cond = (face_scales >= self.gsu_range[0]) & (face_scales <= self.gsu_range[1])
        gt_ignore_box_ids, = torch.where(lower_ignore_cond | upper_ignore_cond)

        rf_centers = (rf_anchors[:,:, [0,1]] + rf_anchors[:,:, [2,3]]) / 2
        rf_norm_value = (rf_anchors[0,0,2] - rf_anchors[0,0,0]) / 2

        # lets match
        for x1,y1,x2,y2 in s_gt_boxes:
            cond_x = (rf_centers[:,:,0] > x1) & (rf_centers[:,:,0] < x2)
            cond_y = (rf_centers[:,:,1] > y1) & (rf_centers[:,:,1] < y2)
            match = cond_x & cond_y
            # match: fh,fw boolean

            # if there is no match than continue
            if match.sum() == 0:
                continue

            # if already has a match than ignore
            ignore_mask[cls_mask & match] = True

            # set as matched
            cls_mask[match] = True

            # set reg targets
            reg_targets[match, 0] = (rf_centers[match, 0] - x1) / rf_norm_value
            reg_targets[match, 1] = (rf_centers[match, 1] - y1) / rf_norm_value
            reg_targets[match, 2] = (rf_centers[match, 0] - x2) / rf_norm_value
            reg_targets[match, 3] = (rf_centers[match, 1] - y2) / rf_norm_value

        return cls_mask,reg_targets,ignore_mask