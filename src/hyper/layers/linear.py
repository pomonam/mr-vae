import torch

from src.config import HyperConfig
from src.hyper.layers.blocks import get_block
from src.hyper.layers.module import HyperModule
import torch.nn as nn


def identity(x):
  return x


class HyperLinear(HyperModule):
  def __init__(self, in_features: int, out_features: int, activation_fnc,
               hyper_config: HyperConfig):
    super().__init__()

    self.in_features = in_features
    self.out_features = out_features
    self.activation_fnc = activation_fnc
    if self.activation_fnc is None:
      self.activation_fnc = identity
    self.hyper_config = hyper_config

    self.linear1 = nn.Linear(in_features, out_features, bias=True)
    if self.hyper_config.include_chunk:
      self.linear2 = nn.Linear(in_features, out_features, bias=True)

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    self.hyper_block_scale1 = get_block("linear")(input_dim, self.out_features)
    self.hyper_block_shift1 = get_block("linear")(input_dim, self.out_features)
    if self.hyper_config.include_chunk:
      self.hyper_block_scale2 = get_block("linear")(input_dim, self.out_features)
      self.hyper_block_shift2 = get_block("linear")(input_dim, self.out_features)
      self.hyper_block_moe = get_block("linear")(input_dim, 2)

    if self.hyper_config.include_layer_norm:
      self.layer_norm1 = torch.nn.LayerNorm(self.out_features,
                                            elementwise_affine=False)
      if self.hyper_config.include_chunk:
        self.layer_norm2 = torch.nn.LayerNorm(self.out_features,
                                              elementwise_affine=False)
    else:
      self.layer_norm1 = None
      self.layer_norm2 = None

  def forward(self, inputs):
    scale1 = self.hyper_block_scale1(self._net_inputs)
    if self.hyper_config.include_sigmoid_activation:
      scale1 = torch.sigmoid(scale1)
    shift1 = self.hyper_block_shift1(self._net_inputs)

    if self.hyper_config.include_chunk:
      scale2 = self.hyper_block_scale2(self._net_inputs)
      if self.hyper_config.include_sigmoid_activation:
        scale2 = torch.sigmoid(scale2)
      shift2 = self.hyper_block_shift2(self._net_inputs)

    act1 = self.linear1(inputs)
    if self.hyper_config.include_chunk:
      act2 = self.linear2(inputs)

    if self.hyper_config.preact_transform:
      sact1 = scale1 * act1 + shift1
      if self.hyper_config.include_chunk:
        sact2 = scale2 * act2 + shift2
        moe = self.hyper_block_moe(self._net_inputs)
        moe = torch.softmax(moe, 1)
        act = sact1 * moe[:, 0].unsqueeze(-1) + sact2 * moe[:, 1].unsqueeze(-1)

        if self.hyper_config.include_residual_connection:
          return self.activation_fnc(act + act1 + act2)
        else:
          return self.activation_fnc(act)

      else:
        if self.hyper_config.include_residual_connection:
          return self.activation_fnc(sact1 + act1)
        else:
          return self.activation_fnc(act1)

    else:
      act1 = self.activation_fnc(act1)
      sact1 = scale1 * act1 + shift1
      if self.hyper_config.include_chunk:
        act2 = self.activation_fnc(act2)
        sact2 = scale2 * act2 + shift2
        moe = self.hyper_block_moe(self._net_inputs)
        moe = torch.softmax(moe, 1)
        act = sact1 * moe[:, 0].unsqueeze(-1) + sact2 * moe[:, 1].unsqueeze(-1)
        if self.hyper_config.include_residual_connection:
          normal = act1 * moe[:, 0].unsqueeze(-1) + act2 * moe[:, 1].unsqueeze(-1)
          return normal + act
        else:
          return act
      else:
        if self.hyper_config.include_residual_connection:
          return act1 + sact1
        else:
          return sact1


    # if self.hyper_config.include_sigmoid_activation == "none":
    #   scale = scale
    # elif self.hyper_config.hyper_activation == "sigmoid":
    #   scale = torch.sigmoid(scale)
    # elif self.hyper_config.hyper_activation == "tanh":
    #   scale = torch.tanh(scale)
    # else:
    #   raise NotImplementedError
    #
    # if self.hyper_config.include_layer_norm:
    #   act = self.layer_norm(inputs)
    # else:
    #   act = inputs
    #
    # act = scale * act + shift
    #
    # if self.hyper_config.include_residual_connection:
    #   return inputs + act
    # else:
    #   return act
