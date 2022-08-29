import torch
import torch.nn as nn


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
