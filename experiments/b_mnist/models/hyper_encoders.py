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


class ResNetEncoder(BaseHyperEncoder):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResNet(1, [64, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten())

    def forward(self, x):
        return self.layers(x)
