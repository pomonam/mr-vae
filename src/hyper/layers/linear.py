import torch
import torch.nn as nn

from src.config import HyperConfig
from src.hyper.layers.blocks import get_block
from src.hyper.layers.module import get_activation
from src.hyper.layers.module import HyperModule


class HyperLinear(HyperModule):

  def __init__(self,
               in_features: int,
               out_features: int,
               activation_fnc: str,
               hyper_config: HyperConfig,
               bias=True):
    super().__init__()

    self.in_features = in_features
    self.out_features = out_features
    self.activation_fnc = get_activation(activation_fnc)
    self.hyper_config = hyper_config
    self.bias = bias

    self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    self.hyper_block_scale = get_block("linear")(input_dim, self.out_features)
    self.hyper_block_shift = get_block("linear")(input_dim, self.out_features)

    if self.hyper_config.include_layer_norm:
      self.layer_norm = torch.nn.LayerNorm(
          self.out_features, elementwise_affine=False)
    else:
      self.layer_norm = nn.Identity()

  def forward(self, inputs):
    scale = self.hyper_block_scale(self._net_inputs)
    if self.hyper_config.include_sigmoid_activation:
      scale = torch.sigmoid(scale)
    shift = self.hyper_block_shift(self._net_inputs)

    if self.hyper_config.preact_transform:
      pre_act_prev = self.linear(inputs)
      pre_act = self.layer_norm(pre_act_prev)
      pre_act = scale * pre_act
      if self.hyper_config.include_shift:
        pre_act = pre_act + shift
      act = self.activation_fnc(pre_act)
      if self.hyper_config.include_residual_connection:
        return act + self.activation_fnc(pre_act_prev)
      else:
        return act
    else:
      pre_act = self.linear(inputs)
      act_prev = self.activation_fnc(pre_act)
      act = self.layer_norm(act_prev)
      act = scale * act
      if self.hyper_config.include_shift:
        act = act + shift
      if self.hyper_config.include_residual_connection:
        return act + act_prev
      else:
        return act
