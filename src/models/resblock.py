import torch
from torch import nn

from src.hyper.layers import get_hyper_layer


class ResBlock(nn.Module):

  def __init__(self, channels: int) -> None:
    super().__init__()

    self.conv_block = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.conv_block(x)


class HyperResBlock(nn.Module):
  """Residual block with paper-style per-conv pre-activation modulation."""

  def __init__(self, channels: int, decoder: bool = False) -> None:
    super().__init__()

    self.conv_block = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        get_hyper_layer(channels, decoder=decoder),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        get_hyper_layer(channels, decoder=decoder),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x + self.conv_block(x)
