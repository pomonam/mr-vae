import math

import torch
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

    if self.hyper_config.include_linear_transformation:
      self.weight = torch.nn.Parameter(torch.empty((self.width, self.width)))
      self.bias = torch.nn.Parameter(torch.empty(self.width))
      self.reset_parameters()

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    self.beta_block_weight = get_block("linear")(input_dim, self.width)
    self.beta_block_bias = get_block("linear")(input_dim, self.width)

    self.beta_block_weight2 = get_block("linear")(input_dim, self.width)
    self.beta_block_bias2 = get_block("linear")(input_dim, self.width)

  def reset_parameters(self) -> None:
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    init.uniform_(self.bias, -bound, bound)

  def forward(self, inputs):
    hyper_weight = self.beta_block_weight(self._net_beta)
    hyper_bias = self.beta_block_bias(self._net_beta)
    hyper_weight2 = self.beta_block_weight2(self._net_beta)
    hyper_bias2 = self.beta_block_bias2(self._net_beta)

    if self.hyper_config.include_sigmoid_activation:
      hyper_weight = torch.sigmoid(hyper_weight)
      hyper_weight2 = torch.sigmoid(hyper_weight2)

    # if self.hyper_config.include_linear_transformation:
    #   out = inputs * hyper_weight + hyper_bias
    #   out = F.linear(out, self.weight, self.bias)
    # else:
    #   # out = inputs + inputs * hyper_weight + hyper_bias
    #   out = inputs * hyper_weight + hyper_bias

    if self.hyper_config.include_linear_transformation:
      out = inputs * hyper_weight2 + hyper_bias2
      out = F.linear(out, self.weight, self.bias)
      out = inputs + inputs * hyper_weight + hyper_bias + out
    else:
      # out = inputs + inputs * hyper_weight + hyper_bias
      out = inputs + inputs * hyper_weight + hyper_bias

    return out
