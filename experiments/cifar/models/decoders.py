import torch
from torch import nn

from src.models.base_decoder import BaseDecoder
from src.models.resnet import ResBlock


class ResNetDecoder(BaseDecoder):

    def __init__(self):
        super().__init__()

        self.first_proj = nn.Linear(64, 128 * 8 * 8)
        layers = nn.Sequential(
          ResBlock(in_channels=128, out_channels=32),
          ResBlock(in_channels=128, out_channels=32),
          nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
          nn.ReLU(),
          nn.ConvTranspose2d(64, 3, 4, 2, padding=1)
        )
        self.layers = layers

    def forward(self, z):
        z = self.first_proj(z)
        z = z.reshape(z.shape[0], 128, 8, 8)
        z = self.layers(z)
        return z
