import torch
import torch.nn as nn
from src.hyper.layer import HyperLayer


class ResBlock(nn.Module):

  def __init__(self, channels):
    nn.Module.__init__(self)

    self.conv_block = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channels),
    )

  def forward(self, x: torch.tensor) -> torch.Tensor:
    return x + self.conv_block(x)


class HyperResBlock(nn.Module):

  def __init__(self, channels, hyper_cfg):
    nn.Module.__init__(self)

    self.conv_block = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        HyperLayer(channels, hyper_cfg),
        nn.BatchNorm2d(channels),
    )

  def forward(self, x: torch.tensor) -> torch.Tensor:
    return x + self.conv_block(x)
