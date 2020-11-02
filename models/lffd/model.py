import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple,List

from utils.matcher import LFFDMatcher

def conv_layer(in_channels:int, out_channels:int,
        kernel_size:int=3, stride:int=1, padding:int=1, bias:bool=True) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=bias),
        nn.ReLU(inplace=False))

class ResBlock(nn.Module):
    def __init__(self, features:int):
        super(ResBlock,self).__init__()
        self.conv1 = conv_layer(features,features)
        self.conv2 = conv_layer(features,features)

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)       # c(i) => c(i+1)
        x = self.conv2(x) + input   # c(i+1) => c(i+2)
        return x

class DetectionHead(nn.Module):
    def __init__(self, features:int, rf_size:int, rf_stride:int,
            lower_scale:int, upper_scale:int, num_classes:int=1, num_target_reg:int=4):

        super(DetectionHead,self).__init__()

        self.rf_size = rf_size
        self.rf_stride = rf_stride

        self.natural_rfs = []

        self.cls_head = nn.Sequential(
            conv_layer(features,features,kernel_size=1,padding=0),
            conv_layer(features,num_classes,kernel_size=1,padding=0))

        self.reg_head = nn.Sequential(
            conv_layer(features,features,kernel_size=1,padding=0),
            conv_layer(features,num_target_reg,kernel_size=1,padding=0))

        self.matcher = LFFDMatcher(lower_scale, upper_scale)

    def gen_rf_centers(self, fmaph:int, fmapw:int, device:str='cpu') -> torch.Tensor:
        """takes feature map h and w and reconstructs rf centers as tensor

        Args:
            fmaph (int): featuremap hight
            fmapw (int): featuremap width

        Returns:
            torch.Tensor: rf centers as fmaph x fmapw x 2 (x, y order)
        """

        # y: fmaph x fmapw
        # x: fmaph x fmapw
        y,x = torch.meshgrid(
            torch.arange(fmaph, dtype=torch.float32), torch.arange(fmapw, dtype=torch.float32))

        # rfs: fmaph x fmapw x 2
        rfs = torch.stack([x,y], dim=-1) + 0.5

        rfs *= self.rf_stride

        return rfs.to(device)

    def forward(self, input:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        pred_cls = self.cls_head(input)
        reg_cls = self.reg_head(input)
        return pred_cls,reg_cls

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


if __name__ == "__main__":
    model = LFFD()
    print(model)