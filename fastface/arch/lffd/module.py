import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from typing import Tuple,List,Dict
from .blocks import (
    LFFDBackboneV1,
    LFFDBackboneV2,
    DetectionHead
)

from cv2 import cv2
import fastface as ff

import math
import numpy as np

class LFFD(nn.Module):
    __CONFIGS__ = {
        "560_25L_8S":{
            "backbone_name": "560_25L_8S",
            'head_infeatures': [64,64,64,64,128,128,128,128],
            'head_outfeatures': [128,128,128,128,128,128,128,128],
            'rf_sizes': [15, 20, 40, 70, 110, 250, 400, 560],
            'rf_start_offsets': [3, 3, 7, 7, 15, 31, 31, 31],
            'rf_strides': [4, 4, 8, 8, 16, 32, 32, 32],
            'scales': [
                (10,15),(15,20),(20,40),(40,70),
                (70,110),(110,250),(250,400),(400,560)
            ] # calculated for 640 image input
        },

        "320_20L_5S":{
            "backbone_name": "320_20L_5S",
            'head_infeatures': [64,64,64,128,128],
            'head_outfeatures': [128,128,128,128,128],
            'rf_sizes': [20, 40, 80, 160, 320],
            'rf_start_offsets': [3, 7, 15, 31, 63],
            'rf_strides': [4, 8, 16, 32, 64],
            'scales': [
                (10,20),(20,40),(40,80),(80,160),(160,320)
            ] # calculated for 640 image input
        }
    }

    _transforms = ff.transform.Compose(
        ff.transform.Interpolate(max_dim=640),
        ff.transform.Padding(target_size=(640,640)),
        ff.transform.Normalize(mean=127.5, std=127.5),
        ff.transform.ToTensor()
    )

    def __init__(self, in_channels:int=3, config:Dict={},
            num_classes:int=1, debug:bool=False, **kwargs):
        super(LFFD,self).__init__()

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

        self.nms = kwargs.get('nms', box_ops.nms)
        self.num_classes = num_classes
        self.__debug = debug

        # TODO check if list lenghts are matched
        if backbone_name == "560_25L_8S":
            self.backbone = LFFDBackboneV1(in_channels)
        elif backbone_name == "320_20L_5S":
            self.backbone = LFFDBackboneV2(in_channels)
        else:
            raise ValueError(f"given backbone name: {backbone_name} is not valid")

        self.heads = nn.ModuleList([
            DetectionHead(idx+1,infeatures,outfeatures, rf_size, rf_start_offset, rf_stride,
                num_classes=num_classes)
            for idx,(infeatures,outfeatures, rf_size, rf_start_offset, rf_stride) in enumerate(zip(
                head_infeatures,head_outfeatures,rf_sizes,rf_start_offsets,rf_strides))
        ])

        self.cls_loss_fn = ff.loss.BinaryCrossEntropy(negative_selection_rule="mix")
        self.reg_loss_fn = ff.loss.L2Loss()

    def forward(self, x:torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits = self.backbone(x)

        cls_logits:List = []
        reg_logits:List = []

        for head,logit in zip(self.heads,logits):
            cls_l,reg_l = head(logit)
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
            boxes = self.heads[i].anchor_box_gen.logits_to_boxes(
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

    def training_step(self, batch:Tuple[torch.Tensor, List],
            batch_idx:int, **hparams) -> torch.Tensor:

        imgs,targets = batch
        device = imgs.device
        dtype = imgs.dtype

        """
        ## targets
        {
            "heads": [
                {
                    "target_cls":       fh',fw'    | torch.float,
                    "ignore_cls_mask":  fh',fw'    | torch.bool,
                    "target_regs":      fh', fw',4 | torch.float,
                    "reg_mask":         fh',fw'    | torch.bool
                }
            ],
            "gt_boxes": torch.Tensor
        }
        """

        num_of_heads = len(self.heads)
        heads_cls_logits,heads_reg_logits = self(imgs)
        head_losses:List = []
        heads = targets['heads']

        for i in range(num_of_heads):
            # *for each head
            target_cls = heads[i]['target_cls'].to(device,dtype)
            ignore_cls_mask = heads[i]['ignore_cls_mask'].to(device)
            target_regs = heads[i]['target_regs'].to(device,dtype)
            reg_mask = heads[i]['reg_mask'].to(device)

            cls_logits = heads_cls_logits[i].permute(0,2,3,1)
            reg_logits = heads_reg_logits[i].permute(0,2,3,1)

            _cls_logits = cls_logits[~ignore_cls_mask].squeeze()
            _target_cls = target_cls[~ignore_cls_mask]

            _reg_logits = reg_logits[reg_mask]
            _target_regs = target_regs[reg_mask]

            cls_loss = self.cls_loss_fn(_cls_logits, _target_cls)
            reg_loss = self.reg_loss_fn(_reg_logits, _target_regs)

            if torch.isnan(cls_loss).any():
                print(cls_loss)
                print(_cls_logits.shape, _target_cls.shape)
                print((_target_cls == 1).sum())

            if torch.isnan(reg_loss).any():
                print(_reg_logits)
                print(_reg_logits.shape, _target_regs.shape)

            head_losses.append( cls_loss + reg_loss )

        return sum(head_losses)

    def validation_step(self, batch:Tuple[torch.Tensor, Dict],
            batch_idx:int, **hparams) -> Dict:

        imgs,targets = batch
        device = imgs.device
        dtype = imgs.dtype
        batch_size = imgs.size(0)

        det_threshold = hparams.get('det_threshold', 0.11)
        iou_threshold = hparams.get('iou_threshold', .4)
        keep_n = hparams.get('keep_n', 10000)

        """
        ## targets
        {
            "heads": [
                {
                    "target_cls":       fh',fw'   | torch.float,
                    "ignore_cls_mask":  fh',fw'   | torch.bool,
                    "target_regs":      fh', fw',4 | torch.float,
                    "reg_mask":         fh',fw'   | torch.bool
                }
            ],
            "gt_boxes": torch.Tensor
        }
        """

        num_of_heads = len(self.heads)
        heads_cls_logits,heads_reg_logits = self(imgs)
        head_losses:List = []
        heads = targets['heads']
        gt_boxes = targets['gt_boxes']

        preds:List = []

        for i in range(num_of_heads):
            # *for each head
            target_cls = heads[i]['target_cls'].to(device,dtype)
            ignore_cls_mask = heads[i]['ignore_cls_mask'].to(device)
            target_regs = heads[i]['target_regs'].to(device,dtype)
            reg_mask = heads[i]['reg_mask'].to(device)

            cls_logits = heads_cls_logits[i].permute(0,2,3,1)
            reg_logits = heads_reg_logits[i].permute(0,2,3,1)

            pred_boxes = self.heads[i].anchor_box_gen.logits_to_boxes(reg_logits)

            scores = torch.sigmoid(cls_logits)
            pred_boxes = torch.cat([pred_boxes,scores], dim=-1).view(batch_size,-1,5)
            # pred_boxes: bs,(fh*fw),5 as xmin,ymin,xmax,ymax,score
            preds.append(pred_boxes)

            _cls_logits = cls_logits[~ignore_cls_mask].squeeze()
            _target_cls = target_cls[~ignore_cls_mask]

            _reg_logits = reg_logits[reg_mask]
            _target_regs = target_regs[reg_mask]
            cls_loss = self.cls_loss_fn(_cls_logits, _target_cls)
            reg_loss = self.reg_loss_fn(_reg_logits, _target_regs)
            if torch.isnan(cls_loss):
                print("class loss is nan: ",_cls_logits, _target_cls)
                exit(0)
            if torch.isnan(reg_loss):
                print("regression loss is nan: ",reg_loss, _target_regs.shape)
                exit(0)
            head_losses.append( cls_loss + reg_loss )

        preds = torch.cat(preds, dim=1)
        pred_boxes:List = []
        for i in range(batch_size):
            selected_boxes = preds[i, preds[i,:,4] > det_threshold, :]
            pick = self.nms(selected_boxes[:, :4], selected_boxes[:, 4], iou_threshold)
            selected_boxes = selected_boxes[pick,:]
            orders = selected_boxes[:, 4].argsort(descending=True)
            pred_boxes.append(selected_boxes[orders,:][:keep_n,:])

        return {
            'loss': sum(head_losses),
            'preds': pred_boxes,
            'gts': gt_boxes
        }

    def test_step(self, batch:Tuple[torch.Tensor, Dict],
            batch_idx:int, **hparams) -> Dict:

        imgs,gt_boxes = batch
        batch_size = imgs.size(0)

        det_threshold = hparams.get('det_threshold', 0.11)
        iou_threshold = hparams.get('iou_threshold', .4)
        keep_n = hparams.get('keep_n', 10000)

        num_of_heads = len(self.heads)

        heads_cls_logits,heads_reg_logits = self(imgs)

        preds:List = []
        for i in range(num_of_heads):
            # *for each head
            cls_logits = heads_cls_logits[i].permute(0,2,3,1)
            reg_logits = heads_reg_logits[i].permute(0,2,3,1)

            pred_boxes = self.heads[i].anchor_box_gen.logits_to_boxes(reg_logits)

            scores = torch.sigmoid(cls_logits)
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
            'gts': gt_boxes
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