import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List,Tuple
import math

class Matcher():
    def __init__(self, rf_start_offsets:List[int], rf_strides:List[int],
        rf_sizes:List[int], scales:List[Tuple[int,int]], fmaps:List[Tuple[int,int]]):
        """[summary]

        Args:
            rf_start_offsets (List[int]): [description]
            rf_strides (List[int]): [description]
            rf_sizes (List[int]): [description]
            scales (List[Tuple[int,int]]): [description]
            fmaps (List[Tuple[int,int]]): List of tuple fmap_h and fmap_w
        """

        # TODO check if all list lenghts are matched

        self.heads:List = []

        for rf_stride, rf_start_offset, rf_size, (lower_scale, upper_scale),fmap in zip(rf_strides,
                rf_start_offsets, rf_sizes, scales, fmaps):
            self.heads.append(
                {
                    "anchor" : AnchorGenerator(rf_stride, rf_start_offset, rf_size),
                    "sl_range": (int(math.floor(lower_scale * 0.9)), lower_scale),
                    "su_range": (upper_scale, int(math.ceil(upper_scale * 1.1))),
                    "fmap": fmap
                }
            )

    def __call__(self, gt_boxes:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """Generates target cls and regs with masks, using ground truth boxes

        Args:
            gt_boxes (torch.Tensor): N',4 as xmin,ymin,xmax,ymax

        Returns:
            List[Tuple[torch.Tensor,torch.Tensor]]:
                (0) target_cls          : fh',fw'   | torch.float
                (1) ignore_cls_mask     : fh',fw'   | torch.bool
                (2) target_regs         : fh',fw',4 | torch.float
                (3) reg_mask            : fh',fw'   | torch.bool
        """
        targets:List = []

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
                targets.append((target_cls, ignore_cls_mask, target_regs, reg_mask))
                continue

            # select max face dim as `face scale` (defined in the paper)
            face_scales,_ = (gt_boxes[:,[2,3]] - gt_boxes[:,[0,1]]).max(dim=1)

            # only select gt boxes that falls between scales
            gt_select_cond = (face_scales >= sl_range[0]) & (face_scales <= su_range[1])

            s_gt_boxes = gt_boxes[gt_select_cond, :]
            face_scales = face_scales[gt_select_cond]

            if s_gt_boxes.size(0) == 0:
                targets.append((target_cls, ignore_cls_mask, target_regs, reg_mask))
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

            targets.append((target_cls, ignore_cls_mask, target_regs, reg_mask))

        return targets

class AnchorGenerator():
    def __init__(self, rf_stride:int, rf_start_offset:int, rf_size:int):
        self.rf_stride = rf_stride
        self.rf_start_offset = rf_start_offset
        self.rf_size = rf_size
        self._dtype = None
        self._device = None
        self._fh = None
        self._fw = None
        self._anchors = None

    def __call__(self, fh:int, fw:int, device:str='cpu',
            dtype:torch.dtype=torch.float32, clip:bool=False) -> torch.Tensor:
        """takes feature map h and w and reconstructs rf anchors as tensor

        Args:
            fh (int): featuremap hight
            fw (int): featuremap width
            device (str, optional): selected device to anchors will be generated. Defaults to 'cpu'.
            dtype (torch.dtype, optional): selected dtype to anchors will be generated. Defaults to torch.float32.
            clip (bool, optional): if True clips regions. Defaults to False.

        Returns:
            torch.Tensor: rf anchors as (fh x fw x 4) (xmin, ymin, xmax, ymax)
        """
        if self._device == device and self._dtype == dtype and self._fh == fh and self._fw == fw:
            return self._anchors.clone()

        self._device = device
        self._dtype = dtype
        self._fh = fh
        self._fw = fw

        # y: fh x fw
        # x: fh x fw
        y,x = torch.meshgrid(
            torch.arange(fh, dtype=dtype, device=device),
            torch.arange(fw, dtype=dtype, device=device)
        )

        # rfs: fh x fw x 2
        rfs = torch.stack([x,y], dim=-1)

        rfs *= self.rf_stride
        rfs += self.rf_start_offset

        # rfs: fh x fw x 2 as x,y
        rfs = rfs.repeat(1,1,2) # fh x fw x 2 => fh x fw x 4
        rfs[:,:,:2] = rfs[:,:,:2] - self.rf_size/2
        rfs[:,:,2:] = rfs[:,:,2:] + self.rf_size/2

        if clip:
            rfs[:,:,[0,2]] = torch.clamp(rfs[:,:,[0,2]],0,fw*self.rf_stride)
            rfs[:,:,[1,3]] = torch.clamp(rfs[:,:,[1,3]],0,fh*self.rf_stride)
        
        self._anchors = rfs.clone()

        return rfs

    def logits_to_boxes(self, reg_logits:torch.Tensor) -> torch.Tensor:
        """Applies bounding box regression using regression logits
        Args:
            reg_logits (torch.Tensor): bs,fh,fw,4

        Returns:
            pred_boxes (torch.Tensor): bs,fh,fw,4 as xmin,ymin,xmax,ymax
        """
        fh,fw = reg_logits.shape[1:3]
        device = reg_logits.device
        dtype = reg_logits.dtype
        
        anchors = self(fh,fw,device,dtype)

        # anchors: fh,fw,4
        rf_normalizer = self.rf_size/2
        assert fh == anchors.size(0)
        assert fw == anchors.size(1)

        rf_centers = (anchors[:,:, :2] + anchors[:,:, 2:]) / 2

        pred_boxes = reg_logits.clone()

        pred_boxes[:, :, :, 0] = rf_centers[:, :, 0] - (rf_normalizer*reg_logits[:, :, :, 0])
        pred_boxes[:, :, :, 1] = rf_centers[:, :, 1] - (rf_normalizer*reg_logits[:, :, :, 1])
        pred_boxes[:, :, :, 2] = rf_centers[:, :, 0] - (rf_normalizer*reg_logits[:, :, :, 2])
        pred_boxes[:, :, :, 3] = rf_centers[:, :, 1] - (rf_normalizer*reg_logits[:, :, :, 3])

        pred_boxes[:,:,:,[0,2]] = torch.clamp(pred_boxes[:,:,:,[0,2]],0,fw*self.rf_stride)
        pred_boxes[:,:,:,[1,3]] = torch.clamp(pred_boxes[:,:,:,[1,3]],0,fh*self.rf_stride)

        return pred_boxes