import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from typing import Tuple,List,Dict
from .blocks import (
    conv3x3,
    ResBlock,
    DetectionHead
)

from mypackage.utils.utils import random_sample_selection
from cv2 import cv2

import math
import numpy as np

class LFFD(nn.Module):
    __CONFIGS__ = {
        "560_25L_8S":{
            'filters': [64,64,64,64,128,128,128,128],
            'rf_sizes': [15, 20, 40, 70, 110, 250, 400, 560],
            'rf_start_offsets': [3, 3, 7, 7, 15, 31, 31, 31],
            'rf_strides': [4,4,8,8,16,32,32,32],
            'scales': [
                (10,15),(15,20),(20,40),(40,70),
                (70,110),(110,250),(250,400),(400,560)
            ], # calculated for 640 image input
            'adapter': {
                'type':'gdrive',
                'key':'models/original_lffd_560_25L_8S.pt',
                'args': ('1uE-_dha8g8akfACES_VsMjwst_b1ir37',),
                'kwargs': {
                    'overwrite': False,
                    'unzip': False,
                    'showsize': True
                }
            }
        }
        # TODO "320_20L_5S"
    }

    def __init__(self, in_channels:int=3, config:Dict={},
            num_classes:int=1, debug:bool=False, **kwargs):
        super(LFFD,self).__init__()

        assert "filters" in config, "`filters` must be defined in the config"
        assert "rf_sizes" in config, "`rf_sizes` must be defined in the config"
        assert "rf_start_offsets" in config, "`rf_start_offsets` must be defined in the config"
        assert "rf_strides" in config, "`rf_strides` must be defined in the config"
        assert "scales" in config, "`scales` must be defined in the config"

        filters = config.get('filters')
        rf_sizes = config.get('rf_sizes')
        rf_start_offsets = config.get('rf_start_offsets')
        rf_strides = config.get('rf_strides')
        scales = config.get('scales')

        self.nms = kwargs.get('nms', box_ops.nms)
        self.num_classes = num_classes
        self.__debug = debug

        # TODO check if list lenghts are matched

        # *tiny part
        self.conv1_dw = conv3x3(in_channels, 64, stride=2, padding=0)
        self.relu1 = nn.ReLU()

        self.conv2_dw = conv3x3(64, 64, stride=2, padding=0)
        self.relu2 = nn.ReLU()

        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(64)
        self.res_block3 = ResBlock(64)
        self.res_block4 = ResBlock(64)

        # *small part
        self.conv3_dw = conv3x3(64, 64, stride=2, padding=0)
        self.relu3 = nn.ReLU()

        self.res_block5 = ResBlock(64)
        self.res_block6 = ResBlock(64)

        # *medium part
        self.conv4_dw = conv3x3(64, 128, stride=2, padding=0)
        self.relu4 = nn.ReLU()
        self.res_block7 = ResBlock(128)

        # *large part
        self.conv5_dw = conv3x3(128, 128, stride=2, padding=0)
        self.relu5 = nn.ReLU()
        self.res_block8 = ResBlock(128)
        self.res_block9 = ResBlock(128)
        self.res_block10 = ResBlock(128)

        self.heads = nn.ModuleList([
            DetectionHead(idx+1,infeatures,128,
                rf_size,rf_start_offset,rf_stride,
                lower_scale,upper_scale,
                num_classes=num_classes)
            for idx,(infeatures, rf_size, rf_start_offset, rf_stride, (lower_scale,upper_scale)) in enumerate(zip(
                filters,rf_sizes,rf_start_offsets,rf_strides,scales))
        ])

    def forward(self, x:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # *tiny part
        c1 = self.conv1_dw(x)         # 3 => 64
        r1 = self.relu1(c1)

        c2 = self.conv2_dw(r1)          # 64 => 64
        r2 = self.relu2(c2)

        r4,c4 = self.res_block1(r2,c2)        # 64 => 64
        r6,c6 = self.res_block2(r4,c4)        # 64 => 64
        r8,c8 = self.res_block3(r6,c6)        # 64 => 64
        r10,_ = self.res_block4(r8,c8)      # 64 => 64

        # *small part
        c11 = self.conv3_dw(r10)       # 64 => 64
        r11 = self.relu3(c11)

        r13,c13 = self.res_block5(r11,c11)      # 64 => 64
        r15,_ = self.res_block6(r13,c13)      # 64 => 64

        # *medium part
        c16 = self.conv4_dw(r15)       # 64 => 128
        r16 = self.relu4(c16)

        r18,_ = self.res_block7(r16,c16)      # 128 => 128

        # *large part
        c19 = self.conv5_dw(r18)       # 128 => 128
        r19 = self.relu5(c19)

        r21,c21 = self.res_block8(r19,c19)      # 128 => 128
        r23,c23 = self.res_block9(r21,c21)      # 128 => 128
        r25,_ = self.res_block10(r23,c23)     # 128 => 128

        cls_logits:List = []
        reg_logits:List = []

        for i,logit in enumerate([r8,r10,r13,r15,r18,r21,r23,r25]):
            cls_l,reg_l = self.heads[i](logit)
            cls_logits.append(cls_l)
            reg_logits.append(reg_l)

        return cls_logits,reg_logits

    def predict(self, x:torch.Tensor, det_threshold:float=.95,
            keep_n:int=10000, iou_threshold:float=.4) -> List[torch.Tensor]:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        cls_logits,reg_logits = self.forward(x)
        preds:List = []
        for i in range(len(self.heads)):
            # *for each head
            fh,fw = cls_logits[i].shape[2:]

            cls_ = cls_logits[i].permute(0,2,3,1).view(batch_size, -1, self.num_classes)
            reg_ = reg_logits[i].permute(0,2,3,1).view(batch_size, -1, 4)

            scores = torch.sigmoid(cls_.view(batch_size,fh,fw,self.num_classes))
            # original positive pred score dim is 0
            boxes = self.heads[i].apply_bbox_regression(
                reg_.view(batch_size,fh,fw,4))

            boxes = torch.cat([boxes,scores], dim=-1).view(batch_size,-1,5)
            # boxes: bs,(fh*fw),5 as xmin,ymin,xmax,ymax,score
            preds.append(boxes)
        preds = torch.cat(preds, dim=1)

        pred_boxes:List = []
        for i in range(batch_size):
            selected_boxes = preds[i, preds[i,:,4] > det_threshold, :]
            pick = box_ops.nms(selected_boxes[:, :4], selected_boxes[:, 4], iou_threshold)
            selected_boxes = selected_boxes[pick,:]
            orders = selected_boxes[:, 4].argsort(descending=True)
            pred_boxes.append(selected_boxes[orders,:][:keep_n,:].cpu())
        return pred_boxes

    def training_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]],
            batch_idx:int, **hparams) -> torch.Tensor:

        imgs,gt_boxes = batch

        device = imgs.device
        dtype = imgs.dtype
        num_of_heads = len(self.heads)

        heads_cls_logits,heads_reg_logits = self(imgs)

        cls_loss:List[torch.Tensor] = []
        reg_loss:List[torch.Tensor] = []

        for head_idx in range(num_of_heads):
            # *for each head
            fh,fw = heads_cls_logits[head_idx].shape[2:]

            target_cls,mask_cls,target_regs,mask_regs = self.heads[head_idx].build_targets_v2(
                (fh,fw), gt_boxes, device=device, dtype=dtype)
            #target_cls          : bs,fh',fw'      | type: model.dtype         | device: model.device
            #mask_cls            : bs,fh',fw'      | type: torch.bool          | device: model.device
            #target_regs         : bs,fh',fw',4    | type: model.dtype         | device: model.device
            #mask_regs           : bs,fh',fw'      | type: torch.bool          | device: model.device

            cls_logits = heads_cls_logits[head_idx].permute(0,2,3,1)
            reg_logits = heads_reg_logits[head_idx].permute(0,2,3,1)

            head_cls_loss,head_reg_loss = self.heads[head_idx].compute_loss(
                (cls_logits,target_cls,mask_cls),
                (reg_logits,target_regs,mask_regs)
            )

            cls_loss.append(head_cls_loss)
            reg_loss.append(head_reg_loss)
        cls_loss = torch.cat(cls_loss).mean()
        reg_loss = torch.cat(reg_loss).mean()
        return cls_loss + reg_loss

    def validation_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]],
            batch_idx:int, **hparams) -> Dict:

        det_threshold = hparams.get('det_threshold', 0.11)
        iou_threshold = hparams.get('iou_threshold', .4)
        keep_n = hparams.get('keep_n', 10000)

        imgs,o_gt_boxes = batch
        gt_boxes = [gt[:,:4] for gt in o_gt_boxes]

        batch_size = imgs.size(0)
        device = imgs.device
        dtype = imgs.dtype
        num_of_heads = len(self.heads)

        heads_cls_logits,heads_reg_logits = self(imgs)

        cls_loss:List[torch.Tensor] = []
        reg_loss:List[torch.Tensor] = []
        preds:List[torch.Tensor] = []

        for head_idx in range(num_of_heads):
            # *for each head
            fh,fw = heads_cls_logits[head_idx].shape[2:]

            target_cls,mask_cls,target_regs,mask_regs = self.heads[head_idx].build_targets_v2(
                (fh,fw), gt_boxes, device=device, dtype=dtype)
            #target_cls          : bs,fh',fw'      | type: model.dtype         | device: model.device
            #mask_cls            : bs,fh',fw'      | type: torch.bool          | device: model.device
            #target_regs         : bs,fh',fw',4    | type: model.dtype         | device: model.device
            #mask_regs           : bs,fh',fw'      | type: torch.bool          | device: model.device

            cls_logits = heads_cls_logits[head_idx].permute(0,2,3,1)
            reg_logits = heads_reg_logits[head_idx].permute(0,2,3,1)

            head_cls_loss,head_reg_loss = self.heads[head_idx].compute_loss(
                (cls_logits,target_cls,mask_cls),
                (reg_logits,target_regs,mask_regs)
            )

            head_boxes = self.heads[head_idx].apply_bbox_regression(reg_logits).view(batch_size,-1,4)
            head_scores = torch.sigmoid(cls_logits).view(batch_size,-1,1)
            head_preds = torch.cat([head_boxes,head_scores],dim=-1)

            cls_loss.append(head_cls_loss)
            reg_loss.append(head_reg_loss)
            preds.append(head_preds)

        cls_loss = torch.cat(cls_loss).mean()
        reg_loss = torch.cat(reg_loss).mean()
        loss = cls_loss + reg_loss

        preds = torch.cat(preds, dim=1)
        pick = preds[:,:,-1] > det_threshold
        selected_preds:List[torch.Tensor] = []
        for head_idx in range(num_of_heads):
            head_preds = preds[head_idx, pick[head_idx], :]
            # TODO test batched nms
            select = self.nms(head_preds[:, :4], head_preds[:, -1], iou_threshold)
            orders = head_preds[select, -1].argsort(descending=True)
            selected_preds.append(head_preds[orders, :][keep_n])

        return {
            'loss':loss,
            'preds': selected_preds,
            'gts': o_gt_boxes
        }

    def test_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]],
            batch_idx:int, **hparams) -> Dict:

        det_threshold = hparams.get('det_threshold', 0.11)
        iou_threshold = hparams.get('iou_threshold', .4)
        keep_n = hparams.get('keep_n', 10000)

        imgs,o_gt_boxes = batch
        if isinstance(imgs, List):
            # TODO handle dynamic batching using max sized image dim
            batch_size = len(imgs)
            assert batch_size == 1,"batch size must be 1 if batch supplied as list"
            cls_logits,reg_logits = self(imgs[0])
        else:
            batch_size = imgs.size(0)
            cls_logits,reg_logits = self(imgs)

        preds:List = []

        for i in range(len(self.heads)):
            # *for each head
            fh,fw = cls_logits[i].shape[2:]
            cls_logits[i] = cls_logits[i].permute(0,2,3,1).view(batch_size, -1)
            reg_logits[i] = reg_logits[i].permute(0,2,3,1).view(batch_size, -1, 4)

            with torch.no_grad():
                # TODO checkout here
                scores = torch.sigmoid(cls_logits[i].view(batch_size, fh, fw, self.num_classes))
                pred_boxes = self.heads[i].apply_bbox_regression(
                    reg_logits[i].view(batch_size,fh,fw,4))

                pred_boxes = torch.cat([pred_boxes,scores], dim=-1).view(batch_size,-1,5)
                # pred_boxes: bs,(fh*fw),5 as xmin,ymin,xmax,ymax,score
                preds.append(pred_boxes)

        preds = torch.cat(preds, dim=1)
        pred_boxes:List = []
        for i in range(batch_size):
            selected_boxes = preds[i, preds[i,:,4] > det_threshold, :]
            pick = self.nms(selected_boxes[:, :4], selected_boxes[:, 4], iou_threshold)
            selected_boxes = selected_boxes[pick,:]
            orders = selected_boxes[:, 4].argsort(descending=True)
            pred_boxes.append(selected_boxes[orders,:][:keep_n,:])

        return {
            'preds': pred_boxes,
            'gts': o_gt_boxes
        }

    def configure_optimizers(self, **hparams):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=hparams.get('learning_rate', 1e-1),
            momentum=hparams.get('momentum', 0.9),
            weight_decay=hparams.get('weight_decay', 1e-5))

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=hparams.get("milestones", [600000, 1000000, 1200000, 1400000]),
            gamma=hparams.get("gamma", 0.1))

        return [optimizer], [lr_scheduler]

def debug_show_rf_matches(tensor_imgs:torch.Tensor,
        head_rfs:List[torch.Tensor], batch_gt_boxes:List[torch.Tensor]):
    imgs = debug_tensor2img(tensor_imgs)
    np_gt_boxes = [gt_boxes.cpu().long().numpy() for gt_boxes in batch_gt_boxes]
    np_head_rfs = [rfs.cpu().long().numpy() for rfs in head_rfs]
    for img,gt_boxes in zip(imgs,np_gt_boxes):
        timg = img.copy()
        for x1,y1,x2,y2 in gt_boxes:
            timg = cv2.rectangle(timg, (x1,y1),(x2,y2),(0,255,0))

        for i,rfs in enumerate(np_head_rfs):
            print(f"for head {i+1}")
            values,pick = box_ops.box_iou(torch.tensor(rfs),torch.tensor(gt_boxes)).max(dim=0)
            pick = pick.cpu().numpy().tolist()
            nimg = timg.copy()
            for j,(x1,y1,x2,y2) in enumerate(rfs[pick,:]):
                print("iou: ",values[j])
                nimg = cv2.rectangle(nimg, (x1,y1), (x2,y2),(0,0,255),2)
            cv2.imshow("",nimg)
            cv2.waitKey(0)

def debug_show_pos_neg_ig_with_gt(tensor_imgs:torch.Tensor,
        batch_gt_boxes:List[torch.Tensor], head_rfs:List[torch.Tensor],
        debug_fmaps:List[Tuple], opos_mask:torch.Tensor,
        oneg_mask:torch.Tensor, oig_mask:torch.Tensor):

    pos_mask = opos_mask.cpu()
    neg_mask = oneg_mask.cpu()
    ig_mask = oig_mask.cpu()

    imgs = debug_tensor2img(tensor_imgs)
    np_gt_boxes = [gt_boxes.cpu().long().numpy() for gt_boxes in batch_gt_boxes]
    np_head_rfs = [rfs.cpu().long().numpy() for rfs in head_rfs]
    for img,gt_boxes,p_mask,n_mask,i_mask in zip(imgs, np_gt_boxes,
            pos_mask, neg_mask, ig_mask):
        # p_mask: 66800,
        print(f"positives: {p_mask.sum()} negatives:{n_mask.sum()} ignore:{i_mask.sum()}")
        timg = img.copy()
        for x1,y1,x2,y2 in gt_boxes:
            timg = cv2.rectangle(timg,(x1,y1),(x2,y2),(255,0,0),2)
        for (fh,fw,start),rfs in zip(debug_fmaps,np_head_rfs):
            # rfs  : 25600,4 as xmin,ymin,xmax,ymax
            crfs = ((rfs[:, [2,3]] + rfs[:, [0,1]]) // 2).astype(np.int32)
            # crfs : 25600,2 as cx,cy
            p_crfs = crfs[p_mask[start: start+fh*fw], :]
            n_crfs = crfs[n_mask[start: start+fh*fw], :]
            i_crfs = crfs[i_mask[start: start+fh*fw], :]

            for cx,cy in p_crfs:
                timg = cv2.circle(timg, (cx,cy), 4, (0,255,0))

            for cx,cy in n_crfs:
                timg = cv2.circle(timg, (cx,cy), 4, (0,0,255))

            for cx,cy in i_crfs:
                timg = cv2.circle(timg, (cx,cy), 4, (0,255,255))

        cv2.imshow("",timg)
        if cv2.waitKey(0) == 27:
            exit(0)

def debug_tensor2img(imgs:torch.Tensor) -> List[np.ndarray]:
    np_imgs = []
    for img in imgs:
        nimg = (img*127.5 + 127.5).permute(1,2,0).cpu().numpy().astype(np.uint8)
        #nimg = cv2.UMat(nimg).get()
        np_imgs.append(nimg)
    return np_imgs