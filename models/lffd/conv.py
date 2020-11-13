import torch.nn as nn

def conv_layer(in_channels:int, out_channels:int,
        kernel_size:int=3, stride:int=1, padding:int=1,
        bias:bool=False) -> nn.Module:

    conv = nn.Conv2d(in_channels, out_channels,
        kernel_size=kernel_size, stride=stride,
        padding=padding, bias=bias)

    nn.init.xavier_normal_(conv.weight)

    if conv.bias is not None:
        conv.bias.data.fill_(0)

    return nn.Sequential(
        conv,
        nn.ReLU6(inplace=False))