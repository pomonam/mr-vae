import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.encoders.base import BaseEncoder


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    "3x3 deconvolution with padding"
    return nn.ConvTranspose2d(in_planes,
                              out_planes,
                              kernel_size=3,
                              stride=stride,
                              padding=1,
                              output_padding=output_padding,
                              bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_planes, planes, strides):
        super(ResNet, self).__init__()
        assert len(planes) == len(strides)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            block = BasicBlock(in_planes, plane, stride=stride)
            blocks.append(block)
            in_planes = plane

        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)
