import torch
from typing import Tuple
import math

# TODO implement RF/Anchor and Ground Truth Matcher

class LFFDMatcher():
    def __init__(self, min_scale:int, max_scale:int, ignore_cls_gray_scale:bool=False, ignore_reg_gray_scale:bool=False):
        # TODO handle multi-class later
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.gsl_range,self.gsu_range = LFFDMatcher.get_gray_scale_range(min_scale,max_scale)
        self.ignore_cls_gray_scale = ignore_cls_gray_scale
        self.ignore_reg_gray_scale = ignore_reg_gray_scale

        # TODO add flag for cls_logit_mask ignoring gray scales or not
        # TODO add flag for reg_logit_mask ignoring gray scales or not

    @staticmethod
    def get_gray_scale_range(sl:int, su:int) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        sl_range = (int(math.floor(sl * 0.9)), sl)
        su_range = (su, int(math.ceil(su * 1.1)))

        return sl_range,su_range

    def __call__(self, rf_anchors:torch.Tensor, gt_boxes:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """Matches given receptive field centers and ground truth boxes

        Args:
            rf_anchors (torch.Tensor): fh,fw,4 as x1,y1,x2,y2
            gt_boxes (torch.Tensor): N,4 as x1,y1,x2,y2

        Returns:
            Tuple:
                torch.Tensor: cls_logit_mask gt box index as fh x fw (ps: -1 means negative >= 0 means positive and -2 means ignored)
                torch.Tensor: reg_logit_mask gt box index as fh x fw (ps: -1 means negative >= 0 means positive and -2 means ignored)
        """
        # TODO fix naming `mask`
        rf_centers = rf_anchors[:,:,:2]
        rf_centers[:, :, 0] = (rf_anchors[:,:,2] + rf_anchors[:,:,0]) / 2
        rf_centers[:, :, 1] = (rf_anchors[:,:,3] + rf_anchors[:,:,1]) / 2

        cls_logit_mask = self._gen_match_mask(rf_centers.clone(), gt_boxes.clone())
        reg_logit_mask = self._gen_match_mask(rf_centers.clone(), gt_boxes.clone())

        return cls_logit_mask,reg_logit_mask

    def _gen_match_mask(self, rf_centers:torch.Tensor, gt_boxes:torch.Tensor) -> torch.Tensor:
        # -1 : negative
        # => 0: gt_box_idx

        fh = rf_centers.size(0)
        fw = rf_centers.size(1)
        rf_centers = rf_centers.reshape(-1,2)
        M = fh*fw

        mask = -1*torch.ones((M,), dtype=torch.int32, device=gt_boxes.device)

        if gt_boxes.size(0) == 0: return mask.view(fh,fw)

        wh = gt_boxes[:,[2,3]] - gt_boxes[:,[0,1]]
        # wh: N,2 as w,h

        areas = (wh[:,0] * wh[:,1]).sqrt()
        # areas: N,

        accepted_box_mask = torch.bitwise_and(areas >= self.min_scale, areas < self.max_scale)
        w_cond = torch.bitwise_and(
            wh[:, 0] > self.gsl_range[1],
            wh[:, 0] < self.gsu_range[0])

        h_cond = torch.bitwise_and(
            wh[:, 1] > self.gsl_range[1],
            wh[:, 1] < self.gsu_range[0])

        accepted_box_mask = torch.bitwise_and(torch.bitwise_and(w_cond,h_cond),accepted_box_mask)


        gt_box_ids = torch.arange(0, gt_boxes.size(0), dtype=torch.int32, device=gt_boxes.device)[accepted_box_mask]
        gt_boxes = gt_boxes[accepted_box_mask]

        # TODO apply ignore here

        N = gt_boxes.size(0)

        if N == 0: return mask.view(fh,fw)

        for i in range(N):
            x1,y1,x2,y2 = gt_boxes[i]
            matches = torch.bitwise_and(
                torch.bitwise_and(rf_centers[:, 0] >= x1, rf_centers[:, 0] <= x2),
                torch.bitwise_and(rf_centers[:, 1] >= y1, rf_centers[:, 1] <= y2)
            )

            not_assigned = mask == -1
            mask[torch.bitwise_and(matches,~not_assigned)] = -2 # ignored
            mask[torch.bitwise_and(matches,not_assigned)] = gt_box_ids[i] # assign the box
        # mask: M, => fh,fw
        return mask.reshape(fh,fw)