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

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    self.hyper_block_scale = get_block("linear")(input_dim, self.width)
    self.hyper_block_shift = get_block("linear")(input_dim, self.width)

    if self.hyper_config.include_layer_norm:
      self.layer_norm = torch.nn.LayerNorm(self.width, elementwise_affine=False)
    else:
      self.layer_norm = None

  def forward(self, inputs):
    scale = self.hyper_block_scale(self._net_beta)
    shift = self.hyper_block_shift(self._net_beta)

    if self.hyper_config.hyper_activation == "none":
      scale = scale
    elif self.hyper_config.hyper_activation == "sigmoid":
      scale = torch.sigmoid(scale)
    elif self.hyper_config.hyper_activation == "tanh":
      scale = torch.tanh(scale)
    else:
      raise NotImplementedError

    if self.hyper_config.include_layer_norm:
      act = self.layer_norm(inputs)
    else:
      act = inputs

    act = scale * act + shift

    if self.hyper_config.include_residual_connection:
      return inputs + act
    else:
      return act
