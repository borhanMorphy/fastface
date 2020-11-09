import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List
from .conv import conv_layer
from .resblock import ResBlock
from .detection import DetectionHead

class LFFD(nn.Module):
    # these configs are for 8 headed lffd detector
    __FILTERS__ = [64,64,64,64,128,128,128,128]
    __RF_SIZES__ = [55,71,111,143,223,383,511,639]
    __RF_STRIDES__ = [4,4,8,8,16,32,32,32]
    __SCALES__ = [(10,15),(15,20),(20,40),(40,70),(70,110),(110,250),(250,400),(400,560)] # calculated for 640 image input

    def __init__(self, in_channels:int=3, filters:List[int]=None,
            rf_sizes:List[int]=None, rf_strides:List[int]=None, scales:List[int]=None):
        super(LFFD,self).__init__()

        if filters is None: filters = LFFD.__FILTERS__
        if rf_sizes is None: rf_sizes = LFFD.__RF_SIZES__
        if rf_strides is None: rf_strides = LFFD.__RF_STRIDES__
        if scales is None: scales = LFFD.__SCALES__

        # TODO check if list lenghts are matched

        # *tiny part
        self.downsample_conv1 = conv_layer(in_channels,64,stride=2,padding=1)
        self.downsample_conv2 = conv_layer(64,64,stride=2,padding=1)
        self.res_block1 = ResBlock(64)
        self.res_block2 = ResBlock(64)
        self.res_block3 = ResBlock(64)
        self.res_block4 = ResBlock(64)

        # *small part
        self.downsample_conv3 = conv_layer(64,64,stride=2,padding=1)
        self.res_block4 = ResBlock(64)
        self.res_block5 = ResBlock(64)

        # *medium part
        self.downsample_conv4 = conv_layer(64,128,stride=2,padding=1)
        self.res_block6 = ResBlock(128)

        # *large part
        self.downsample_conv5 = conv_layer(128,128,stride=2,padding=1)
        self.res_block7 = ResBlock(128)
        self.res_block8 = ResBlock(128)
        self.res_block9 = ResBlock(128)

        self.heads = nn.ModuleList([
            DetectionHead(num_of_filters,rf_size,rf_stride,lower_scale,upper_scale)
            for num_of_filters, rf_size, rf_stride, (lower_scale,upper_scale) in zip(filters,rf_sizes,rf_strides,scales)
        ])

    def forward(self, input:torch.Tensor) -> List[torch.Tensor]:
        branch_in:List = []
        # *tiny part
        input = self.downsample_conv1(input)
        input = self.downsample_conv2(input)
        input = self.res_block1(input)
        input = self.res_block2(input)
        input = self.res_block3(input)
        branch_in.append(input.clone())
        input = self.res_block4(input)
        branch_in.append(input.clone())

        # *small part
        input = self.downsample_conv3(input)
        input = self.res_block4(input)
        branch_in.append(input.clone())
        input = self.res_block5(input)
        branch_in.append(input.clone())

        # *medium part
        input = self.downsample_conv4(input)
        input = self.res_block6(input)
        branch_in.append(input.clone())

        # *large part
        input = self.downsample_conv5(input)
        input = self.res_block7(input)
        branch_in.append(input.clone())
        input = self.res_block8(input)
        branch_in.append(input.clone())
        input = self.res_block9(input)
        branch_in.append(input.clone())

        # *heads forward
        branch_out:List = []
        for i in range(len(branch_in)):
            branch_out.append( self.heads[i]( branch_in[i] ) )

        return branch_out

    def training_step(self, batch:Tuple[torch.Tensor, List[torch.Tensor]], batch_idx:int):

        imgs, gt_boxes = batch
        logits = self(imgs)

        loss = 0

        for i in range(len(logits)):
            cls_logits, reg_logits, cls_targets, reg_targets = self.heads[i].build_targets(logits[i], gt_boxes)

            # TODO apply online hard negative mining instead of random selection
            # positive/negative ratio is 1:10
            ratio = 10

            pos_mask = (cls_targets == 1).squeeze()
            select_n = pos_mask.sum() * ratio
            if select_n == 0:
                continue
            mask = torch.zeros((cls_targets.size(0)), dtype=torch.bool, device=cls_targets.device)
            population, = torch.where(~pos_mask)

            selections = random_sample_selection(population.cpu().numpy().tolist(), min(select_n,population.size(0)))
            mask[selections] = True
            mask[pos_mask] = True

            cls_loss = F.binary_cross_entropy_with_logits(cls_logits[mask], cls_targets[mask])
            reg_loss = F.mse_loss(reg_logits,reg_targets) # L2 loss

            loss += ((cls_loss + reg_loss) / 2)

        return loss


if __name__ == "__main__":
    model = LFFD()
    print(model)