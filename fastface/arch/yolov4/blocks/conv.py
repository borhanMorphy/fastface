import torch.nn as nn

def conv3x3(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Module:
    return _conv(3, in_channels, out_channels, stride=1,
        padding=1, normalize=normalize)

def conv3x3_ds(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Module:
    return _conv(3, in_channels, out_channels, stride=2,
        padding=1, normalize=normalize)

def conv1x1(in_channels: int, out_channels: int, normalize: bool = True) -> nn.Module:
    return _conv(1, in_channels, out_channels, stride=1,
        padding=1, normalize=normalize)

def _conv(kernel_size: int, in_channels: int, out_channels: int, bias: bool = False,
        stride: int = 1, padding: int = 1, normalize: bool = True):
    conv_block = nn.Sequential()

    conv_block.add_module('conv', nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias))

    if normalize:
        conv_block.add_module('bn', nn.BatchNorm2d(out_channels))

    conv_block.add_module('leaky', nn.LeakyReLU(
        negative_slope=0.1, inplace=True))

    return conv_block