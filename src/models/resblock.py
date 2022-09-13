import torch
from torch import nn

from src.config import HyperConfig
from src.hyper.layers import get_hyper_layer


class ResBlock(nn.Module):

  def __init__(self, channels: int) -> None:
    super().__init__()

    self.conv_block = nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
    )

  def forward(self, x: torch.tensor) -> torch.Tensor:
    return x + self.conv_block(x)


class HyperResBlock(nn.Module):

  def __init__(self, channels: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    if hyper_cfg.param_type in ["pre_bn", "post_bn"]:
      self.conv_block = nn.Sequential(
          nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
          get_hyper_layer(channels, hyper_cfg),
          nn.ReLU(),
          nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
          get_hyper_layer(channels, hyper_cfg),
      )
    elif hyper_cfg.param_type == "post_act":
      self.conv_block = nn.Sequential(
          nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          get_hyper_layer(channels, hyper_cfg),
          nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
      )
    else:
      raise NotImplementedError

  def forward(self, x: torch.tensor) -> torch.Tensor:
    return x + self.conv_block(x)
