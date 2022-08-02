import math

import torch
from torch import nn

from src.models.base_encoder import BaseEncoder
from src.models.resnet import ResNet


class MLPEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 256)

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        return x


class CNNEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32 * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32 * 2, 32 * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32 * 4, 32 * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)


class ResNetEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResNet(1, [64, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)

