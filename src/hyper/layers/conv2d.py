import math

import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F

from src.config import HyperConfig
from src.hyper.layers.blocks import get_block
from src.hyper.layers.module import get_activation
from src.hyper.layers.module import HyperModule


class HyperConv2d(HyperModule):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               activation_fnc,
               hyper_config: HyperConfig,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               apply_bn=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.bias = bias
    self.activation_fnc = get_activation(activation_fnc)
    self.hyper_config = hyper_config
    self.apply_bn = apply_bn

    self.conv2d = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=self.dilation,
        groups=self.groups,
        bias=bias)

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    self.hyper_block_scale = get_block("linear")(input_dim, self.out_channels)
    self.hyper_block_shift = get_block("linear")(input_dim, self.out_channels)

    if self.hyper_config.include_layer_norm:
      self.layer_norm = torch.nn.GroupNorm(1, self.out_channels, affine=False)
    else:
      self.layer_norm = nn.Identity()

    if self.apply_bn:
      self.batch_norm = nn.BatchNorm2d(self.out_channels)
    else:
      self.batch_norm = nn.Identity()

  def forward(self, inputs):
    scale = self.hyper_block_scale(self._net_inputs).unsqueeze(-1).unsqueeze(-1)
    if self.hyper_config.include_sigmoid_activation:
      scale = torch.sigmoid(scale)
    shift = self.hyper_block_shift(self._net_inputs).unsqueeze(-1).unsqueeze(-1)

    if self.hyper_config.preact_transform:
      pre_act_prev = self.conv2d(inputs)
      pre_act = self.layer_norm(pre_act_prev)
      pre_act = scale * pre_act
      if self.hyper_config.include_shift:
        pre_act = pre_act + shift
      act = self.batch_norm(pre_act)
      act = self.activation_fnc(act)

      if self.hyper_config.include_residual_connection:
        return act + pre_act_prev
      else:
        return act

    else:
      act_prev = self.conv2d(inputs)
      act_prev = self.activation_fnc(self.batch_norm(act_prev))

      act = self.layer_norm(act_prev)
      act = scale * act
      if self.hyper_config.include_shift:
        act = act + shift
      if self.hyper_config.include_residual_connection:
        return act + act_prev
      else:
        return act

    #
    # if self.hyper_config.preact_transform:
    #     if self.hyper_config.include_layer_norm:
    #         hyper_pre_act = scale * self.layer_norm(pre_act) + shift
    #     else:
    #         hyper_pre_act = scale * pre_act + shift
    #
    #     if self.hyper_config.include_chunk:
    #         if self.hyper_config.include_layer_norm:
    #             chunk_hyper_pre_act = chunk_scale * self.layer_norm(chunk_pre_act) + chunk_shift
    #         else:
    #             chunk_hyper_pre_act = chunk_scale * chunk_pre_act + chunk_shift
    #
    #         chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
    #         hyper_pre_act = hyper_pre_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) +\
    #                             chunk_hyper_pre_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #         pre_act = pre_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
    #                     chunk_pre_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #
    #     if self.hyper_config.include_residual_connection:
    #         if self.bn:
    #             return self.activation_fnc(self.batch_norm(pre_act + hyper_pre_act))
    #         else:
    #             return self.activation_fnc(pre_act + hyper_pre_act)
    #     else:
    #         if self.bn:
    #             return self.activation_fnc(self.batch_norm(hyper_pre_act))
    #         else:
    #             return self.activation_fnc(hyper_pre_act)
    #
    # else:
    #     if self.bn:
    #         pre_act = self.batch_norm(pre_act)
    #     act = self.activation_fnc(pre_act)
    #
    #     if self.hyper_config.include_layer_norm:
    #         hyper_act = scale * self.layer_norm(act) + shift
    #     else:
    #         hyper_act = scale * act + shift
    #
    #     if self.hyper_config.include_chunk:
    #         if self.bn:
    #             chunk_pre_act = self.chunk_batch_norm(chunk_pre_act)
    #         chunk_act = self.activation_fnc(chunk_pre_act)
    #         if self.hyper_config.include_layer_norm:
    #             chunk_hyper_act = chunk_scale * self.layer_norm(chunk_act) + chunk_shift
    #         else:
    #             chunk_hyper_act = chunk_scale * chunk_act + chunk_shift
    #
    #         chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
    #         hyper_act = hyper_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
    #                         chunk_hyper_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #         act = act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
    #                 chunk_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    #     if self.hyper_config.include_residual_connection:
    #         return act + hyper_act
    #     else:
    #         return hyper_act
    #
    # # else:
    # # # out = inputs + inputs * hyper_weight + hyper_bias
    # # out = inputs + inputs * hyper_weight.unsqueeze(-1).unsqueeze(
    # #     -1) + hyper_bias.unsqueeze(-1).unsqueeze(-1)
    # #
    # # return out


class HyperConvTranspose2d(HyperModule):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               activation_fnc,
               hyper_config: HyperConfig,
               stride=1,
               padding=0,
               output_padding=0,
               dilation=1,
               groups=1,
               bias=True,
               apply_bn=False):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.bias = bias
    self.activation_fnc = get_activation(activation_fnc)
    self.hyper_config = hyper_config
    self.apply_bn = apply_bn

    self.conv2d = nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=self.dilation,
        groups=self.groups,
        bias=bias)

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    self.hyper_block_scale = get_block("linear")(input_dim, self.out_channels)
    self.hyper_block_shift = get_block("linear")(input_dim, self.out_channels)

    if self.hyper_config.include_layer_norm:
      self.layer_norm = torch.nn.GroupNorm(1, self.out_channels, affine=False)
    else:
      self.layer_norm = nn.Identity()

    if self.apply_bn:
      self.batch_norm = nn.BatchNorm2d(self.out_channels)
    else:
      self.batch_norm = nn.Identity()

  def forward(self, inputs):
    scale = self.hyper_block_scale(self._net_inputs).unsqueeze(-1).unsqueeze(-1)
    if self.hyper_config.include_sigmoid_activation:
      scale = torch.sigmoid(scale)
    shift = self.hyper_block_shift(self._net_inputs).unsqueeze(-1).unsqueeze(-1)

    if self.hyper_config.preact_transform:
      pre_act_prev = self.conv2d(inputs)
      pre_act = self.layer_norm(pre_act_prev)
      pre_act = scale * pre_act
      if self.hyper_config.include_shift:
        pre_act = pre_act + shift
      act = self.batch_norm(pre_act)
      act = self.activation_fnc(act)

      if self.hyper_config.include_residual_connection:
        return act + pre_act_prev
      else:
        return act

    else:
      act_prev = self.conv2d(inputs)
      act_prev = self.activation_fnc(self.batch_norm(act_prev))

      act = self.layer_norm(act_prev)
      act = scale * act
      if self.hyper_config.include_shift:
        act = act + shift
      if self.hyper_config.include_residual_connection:
        return act + act_prev
      else:
        return act
