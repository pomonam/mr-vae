import math

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from src.config import HyperConfig
from src.hyper.layers.blocks import get_block
from src.hyper.layers.module import HyperModule


class HyperLinear(HyperModule):
  def __init__(self, width: int, hyper_config: HyperConfig):
    super().__init__()

    self.width = width
    self.hyper_config = hyper_config

    self.weight = torch.nn.Parameter(torch.empty((self.width, self.width)))
    self.bias = torch.nn.Parameter(torch.empty(self.width))
    self.reset_parameters()

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    block_type = "linear" if hyper_config.preprocess_beta else self.cfg.block_type
    self.beta_block = get_block(block_type)(input_dim, self.width * 2)

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(self.bias, -bound, bound)

  def forward(self, inputs):
    hyper_out = self.beta_block(self._net_beta["net_beta"])
    hyper_weight = hyper_out[:, :self.width]
    hyper_bias = hyper_out[:, self.width:]

    if self.hyper_config.include_sigmoid_activation:
      hyper_bias = torch.sigmoid(hyper_weight)

    out = F.linear(inputs, self.weight, self.bias)
    out = out * hyper_weight + hyper_bias

    return out
