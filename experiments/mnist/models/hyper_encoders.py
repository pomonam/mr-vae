import math

import torch
from torch import nn
from torch.nn import functional as F

from src.hyper.layers.conv2d import HyperConv2d
from src.hyper.layers.linear import HyperLinear
from src.hyper.models import BaseHyperEncoder
from src.models.resnet import ResNet


class HyperMLPEncoder(BaseHyperEncoder):

    def __init__(self, hyper_config):
        super().__init__()

        self.hyper_config = hyper_config
        self.linear1 = HyperLinear(784, 512, "relu", hyper_config)
        self.linear2 = HyperLinear(512, 512, "relu", hyper_config)
        self.linear3 = HyperLinear(512, 256, "relu", hyper_config)

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class HyperCNNEncoder(BaseHyperEncoder):

    def __init__(self, hyper_config):
        super().__init__()

        self.hyper_config = hyper_config

        self.layers = nn.Sequential(
            # nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            HyperConv2d(1, 32, kernel_size=4, stride=2, padding=1, hyper_config=hyper_config, activation_fnc="relu"),
            # nn.ReLU(),
            HyperConv2d(32, 32 * 2, kernel_size=4, stride=2, padding=1, hyper_config=hyper_config, activation_fnc="relu"),
            # HyperConv2d(32 * 2, hyper_config),
            # nn.ReLU(),
            HyperConv2d(32 * 2, 32 * 4, kernel_size=4, stride=2, padding=1, hyper_config=hyper_config,
                        activation_fnc="relu"),
            # HyperConv2d(32 * 4, hyper_config),
            # nn.ReLU(),
            HyperConv2d(32 * 4, 32 * 8, kernel_size=4, stride=2, padding=1, hyper_config=hyper_config, activation_fnc="relu"),
            # HyperConv2d(32 * 8, hyper_config),
            # nn.ReLU(),
            nn.Flatten())
        # else:
        #     self.layers = nn.Sequential(
        #         nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         HyperConv2d(32, hyper_config),
        #         nn.Conv2d(32, 32 * 2, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         HyperConv2d(32 * 2, hyper_config),
        #         nn.Conv2d(32 * 2, 32 * 4, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         HyperConv2d(32 * 4, hyper_config),
        #         nn.Conv2d(32 * 4, 32 * 8, kernel_size=4, stride=2, padding=1),
        #         nn.ReLU(),
        #         HyperConv2d(32 * 8, hyper_config),
        #         nn.Flatten())

    def forward(self, x):
        return self.layers(x)


class HyperBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, hyper_config=None):
        super().__init__()

        self.conv1 = HyperConv2d(
            in_planes,
            planes,
            activation_fnc="relu",
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            bn=True,
            hyper_config=hyper_config)

        self.conv2 = HyperConv2d(
            planes, planes, activation_fnc="none", kernel_size=3, stride=1, padding=1, bias=False,
            hyper_config=hyper_config, bn=True)
        # self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                HyperConv2d(
                    in_planes,
                    self.expansion * planes,
                    activation_fnc="none",
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    bn=True,
                    hyper_config=hyper_config),
                # nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HyperResNet(nn.Module):

    def __init__(self, in_planes, planes, strides, hyper_config):
        super(HyperResNet, self).__init__()
        assert len(planes) == len(strides)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            block = HyperBasicBlock(in_planes, plane, stride=stride, hyper_config=hyper_config)
            blocks.append(block)
            in_planes = plane

        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class HyperResNetEncoder(BaseHyperEncoder):

    def __init__(self, hyper_config):
        super().__init__()
        self.layers = nn.Sequential(
            HyperResNet(1, [64, 64, 64], [2, 2, 2], hyper_config),
            # nn.Conv2d(64, 256, kernel_size=4, stride=1, padding=0, bias=False),
            HyperConv2d(64, 256, activation_fnc="relu", kernel_size=4, stride=1, padding=0,
                        hyper_config=hyper_config, bn=True),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.Flatten())

    def forward(self, x):
        return self.layers(x)
