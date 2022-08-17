import math

import torch
from torch import nn

from src.models.base_encoder import BaseEncoder
from src.models.resnet import ResBlock


class ResNetEncoder(BaseEncoder):

    def __init__(self):
        super().__init__()

        layers = nn.Sequential(
          nn.Conv2d(3, 64, 4, 2, padding=1),
          nn.ReLU(),
          nn.Conv2d(64, 128, 4, 2, padding=1),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, 1, padding=1),
          ResBlock(in_channels=128, out_channels=32),
          ResBlock(in_channels=128, out_channels=32)
        )
        self.layers = layers

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        return x
