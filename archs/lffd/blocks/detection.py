import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List
from .conv import conv1x1
import math
from utils.utils import random_sample_selection
from utils.matcher import LFFDMatcher

class DetectionHead(nn.Module):
    def __init__(self, head_idx:int, infeatures:int,
            features:int, rf_size:int, rf_start_offset:int, rf_stride:int,
            lower_scale:int, upper_scale:int, num_classes:int=1):

        super(DetectionHead,self).__init__()
        self.head_idx = head_idx
        self.rf_size = rf_size
        self.rf_start_offset = rf_start_offset
        self.rf_stride = rf_stride
        self.num_classes = num_classes

        self.det_conv = nn.Sequential(
            conv1x1(infeatures, features), nn.ReLU())

        self.cls_head = nn.Sequential(
            conv1x1(features, features),
            nn.ReLU(),
            conv1x1(features, self.num_classes))

        self.reg_head = nn.Sequential(
            conv1x1(features, features),
            nn.ReLU(),
            conv1x1(features, 4))

        def conv_xavier_init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(conv_xavier_init)

        self.sl_range = (int(math.floor(lower_scale * 0.9)), lower_scale)
        self.su_range = (upper_scale, int(math.ceil(upper_scale * 1.1)))
        self.matcher = LFFDMatcher(lower_scale,upper_scale)

        self.anchors = self.gen_rf_anchors(30,30, device='cpu', dtype=torch.float32)
        self._cached_fh = 30
        self._cached_fw = 30

    def apply_bbox_regression(self, reg_logits:torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,fh,fw,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        fh,fw = reg_logits.shape[1:3]

        rf_anchors = self.gen_rf_anchors(fh,fw,
            device=reg_logits.device,
            dtype=reg_logits.dtype)
        # rf_anchors: fh,fw,4
        rf_normalizer = self.rf_size/2
        assert fh == rf_anchors.size(0)
        assert fw == rf_anchors.size(1)

        rf_centers = (rf_anchors[:,:, :2] + rf_anchors[:,:, 2:]) / 2

        pred_boxes = reg_logits.clone()

        pred_boxes[:, :, :, 0] = rf_centers[:, :, 0] - (rf_normalizer*reg_logits[:, :, :, 0])
        pred_boxes[:, :, :, 1] = rf_centers[:, :, 1] - (rf_normalizer*reg_logits[:, :, :, 1])
        pred_boxes[:, :, :, 2] = rf_centers[:, :, 0] - (rf_normalizer*reg_logits[:, :, :, 2])
        pred_boxes[:, :, :, 3] = rf_centers[:, :, 1] - (rf_normalizer*reg_logits[:, :, :, 3])

        pred_boxes[:,:,:,[0,2]] = torch.clamp(pred_boxes[:,:,:,[0,2]],0,fw*self.rf_stride)
        pred_boxes[:,:,:,[1,3]] = torch.clamp(pred_boxes[:,:,:,[1,3]],0,fh*self.rf_stride)

        return pred_boxes

    def gen_rf_anchors(self, fmaph:int, fmapw:int, device:str='cpu',
            dtype:torch.dtype=torch.float32, clip:bool=False) -> torch.Tensor:
        """takes feature map h and w and reconstructs rf anchors as tensor

        Args:
            fmaph (int): featuremap hight
            fmapw (int): featuremap width
            device (str, optional): selected device to anchors will be generated. Defaults to 'cpu'.
            dtype (torch.dtype, optional): selected dtype to anchors will be generated. Defaults to torch.float32.
            clip (bool, optional): if True clips regions. Defaults to False.

        Returns:
            torch.Tensor: rf anchors as fmaph x fmapw x 4 (xmin, ymin, xmax, ymax)
        """

        # y: fmaph x fmapw
        # x: fmaph x fmapw
        y,x = torch.meshgrid(
            torch.arange(fmaph, dtype=dtype, device=device),
            torch.arange(fmapw, dtype=dtype, device=device)
        )

        # rfs: fmaph x fmapw x 2
        rfs = torch.stack([x,y], dim=-1)

        rfs *= self.rf_stride
        rfs += self.rf_start_offset

        # rfs: fh x fw x 2 as x,y
        rfs = rfs.repeat(1,1,2) # fh x fw x 2 => fh x fw x 4
        rfs[:,:,:2] = rfs[:,:,:2] - self.rf_size/2
        rfs[:,:,2:] = rfs[:,:,2:] + self.rf_size/2

        if clip:
            rfs[:,:,[0,2]] = torch.clamp(rfs[:,:,[0,2]],0,fmapw*self.rf_stride)
            rfs[:,:,[1,3]] = torch.clamp(rfs[:,:,[1,3]],0,fmaph*self.rf_stride)

        return rfs

    def build_targets(self, fmap:Tuple[int,int], gt_boxes:List[torch.Tensor], device:str='cpu',
            dtype:torch.dtype=torch.float32) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """[summary]

        Args:
            fmap (Tuple[int,int]): fmap_h and fmap_w
            gt_boxes (List[torch.Tensor]): [N',4] as xmin,ymin,xmax,ymax
            device (str, optional): target device. Defaults to 'cpu'.
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]: [description]
        """
        # t_cls          : bs,fh',fw'      | type: model.dtype          | device: model.device
        # t_regs         : bs,fh',fw',4    | type: model.dtype          | device: model.device
        # ig             : bs,fh',fw'      | type: torch.bool           | device: model.device
        batch_size = len(gt_boxes)
        fh,fw = fmap

        t_cls = torch.zeros(*(batch_size,fh,fw), dtype=dtype, device=device)
        t_regs = torch.zeros(*(batch_size,fh,fw,4), dtype=dtype, device=device)
        ignore = torch.zeros(*(batch_size,fh,fw), dtype=torch.bool, device=device)

        # TODO cache rf anchors
        rf_anchors = self.gen_rf_anchors(fh, fw, device=device, dtype=dtype)

        # rf_anchors: fh x fw x 4 as xmin,ymin,xmax,ymax
        for i in range(batch_size):
            cls_mask,reg_targets,ignore_mask = self.matcher(rf_anchors,
                gt_boxes[i], self.rf_size/2, device=device, dtype=dtype)

            t_cls[i, cls_mask] = 1

            t_regs[i, :,:,:] = reg_targets
            ignore[i, :,:] = ignore_mask

        return t_cls,t_regs,ignore

    def build_targets_v2(self, fmap:Tuple[int,int], batch_gt_boxes:List[torch.Tensor], device:str='cpu',
            dtype:torch.dtype=torch.float32) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        """[summary]

        Args:
            fmap (Tuple[int,int]): fmap_h and fmap_w
            batch_gt_boxes (List[torch.Tensor]): [N',4] as xmin,ymin,xmax,ymax
            device (str, optional): target device. Defaults to 'cpu'.
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
                (0) target_cls          : bs,fh',fw'      | type: model.dtype         | device: model.device
                (1) mask_cls            : bs,fh',fw'      | type: torch.bool          | device: model.device
                (2) target_regs         : bs,fh',fw',4    | type: model.dtype         | device: model.device
                (3) mask_regs           : bs,fh',fw'      | type: torch.bool          | device: model.device
        """
        batch_size = len(batch_gt_boxes)
        fh,fw = fmap

        target_cls = torch.zeros(*(batch_size,fh,fw), dtype=dtype, device=device)    # 0: bg  | 1: match
        mask_cls = torch.ones(*(batch_size,fh,fw), dtype=torch.bool, device=device)  # False: ignore | True: accept
        target_regs = torch.zeros(*(batch_size,fh,fw,4), dtype=dtype, device=device)
        mask_regs = torch.zeros(*(batch_size,fh,fw), dtype=torch.bool, device=device) # False: ignore | True: accept

        if fh != self._cached_fh or fw != self._cached_fw or device != self.anchors.device or dtype != self.anchors.dtype:
            self.anchors = self.gen_rf_anchors(fh, fw, device=device, dtype=dtype)
            self._cached_fh = fh
            self._cached_fw = fw

        rf_normalizer = self.rf_size/2
        rf_centers = (self.anchors[:,:, [0,1]] + self.anchors[:,:, [2,3]]) / 2
        # rf_anchors: fh x fw x 4 as xmin,ymin,xmax,ymax
        for i in range(batch_size):
            gt_boxes = batch_gt_boxes[i]
            if gt_boxes.size(0) == 0: continue

            # select max face dim as `face scale` (defined in the paper)
            face_scales,_ = (gt_boxes[:,[2,3]] - gt_boxes[:,[0,1]]).max(dim=1)

            # only select gt boxes that falls between scales
            gt_select_cond = (face_scales >= self.sl_range[0]) & (face_scales <= self.su_range[1])

            s_gt_boxes = gt_boxes[gt_select_cond, :]
            face_scales = face_scales[gt_select_cond]

            if s_gt_boxes.size(0) == 0: continue

            # lookup ignores
            lower_ignore_cond = (face_scales >= self.sl_range[0]) & (face_scales <= self.sl_range[1])
            upper_ignore_cond = (face_scales >= self.su_range[0]) & (face_scales <= self.su_range[1])
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
                    mask_cls[i, match] = False
                    continue

                # set as matched
                target_cls[i, match] += 1

                # set reg targets)

                target_regs[i, match, [0]] = (rf_centers[match, [0]] - x1) / rf_normalizer
                target_regs[i, match, [1]] = (rf_centers[match, [1]] - y1) / rf_normalizer
                target_regs[i, match, [2]] = (rf_centers[match, [0]] - x2) / rf_normalizer
                target_regs[i, match, [3]] = (rf_centers[match, [1]] - y2) / rf_normalizer

        multi_matches = target_cls > 1
        positive_matches = target_cls == 1

        mask_cls[ multi_matches ] = False
        target_cls[multi_matches] = 1
        mask_regs[positive_matches] = True

        return target_cls, mask_cls, target_regs, mask_regs

    def compute_loss(self, cls_items:Tuple[torch.Tensor,torch.Tensor,torch.Tensor],
            reg_items:Tuple[torch.Tensor,torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor,torch.Tensor]:
        # (cls_logits,target_cls,mask_cls),
        # (reg_logits,target_regs,mask_regs)
        cls_logits,target_cls,mask_cls = cls_items
        reg_logits,target_regs,mask_regs = reg_items

        device = reg_logits.device
        dtype = reg_logits.dtype

        bs,fh,fw = cls_logits.shape[:3]

        pos_mask = mask_cls & (target_cls == 1)
        neg_mask = mask_cls & (target_cls == 0)
        top_neg_mask = neg_mask.clone().view(-1)
        top_neg_mask[:] = False
        rand_neg_mask = neg_mask.clone().view(-1)

        positives = pos_mask.sum()
        negatives = neg_mask.sum()
        hnm_ratio = 5
        random_ratio = 5
        neg_select_ratio = 0.1

        if positives > 0:
            top_negatives = min(positives*hnm_ratio, negatives//2)
            rand_negatives = min(positives*random_ratio, negatives//2)
        else:
            top_negatives = min(int((pos_mask.view(-1).size(0) // 2) * neg_select_ratio), negatives//2)
            rand_negatives = min(int((pos_mask.view(-1).size(0) // 2) * neg_select_ratio), negatives//2)

        _,top_neg_ids = cls_logits.view(-1).topk(top_negatives)
        top_neg_mask[top_neg_ids] = True
        top_neg_mask = top_neg_mask.view(bs,fh,fw)
        rand_neg_mask[top_neg_ids] = False
        rand_negatives = min(rand_negatives,rand_neg_mask.sum())

        rand_neg_ids, = torch.where(rand_neg_mask)
        pick = random_sample_selection(rand_neg_ids.cpu().numpy().tolist(), rand_negatives)
        rand_neg_mask[:] = False
        rand_neg_mask[pick] = True
        rand_neg_mask = rand_neg_mask.view(bs,fh,fw)
        neg_mask = rand_neg_mask | top_neg_mask

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits[pos_mask | neg_mask].squeeze(), target_cls[pos_mask | neg_mask], reduction='none')

        if mask_regs.sum() > 0:
            reg_loss = F.mse_loss(
                reg_logits[mask_regs, :], target_regs[mask_regs, :], reduction='none')
        else:
            reg_loss = torch.tensor([[0,0,0,0]], dtype=dtype, device=device, requires_grad=True)

        return cls_loss,reg_loss

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.det_conv(x)

        cls_logits = self.cls_head(data)
        # (b,c,h,w)
        reg_logits = self.reg_head(data)
        # (b,c,h,w)
        return cls_logits,reg_logits