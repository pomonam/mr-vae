import math

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from src.config import HyperConfig
from src.hyper.layers.blocks import get_block
from src.hyper.layers.module import HyperModule


class HyperLinear(HyperModule):
  def __init__(self, module: nn.Module, hyper_config: HyperConfig):
    super().__init__(module, hyper_config)

    assert isinstance(module, nn.Linear)
    self.in_features = module.in_features
    self.out_features = module.out_features

    self.weight = torch.nn.Parameter(torch.empty((self.out_features, self.in_features)))
    if module.bias is not None:
      self.bias = torch.nn.Parameter(torch.empty(self.out_features))
    else:
      self.register_parameter("bias", None)
    self.reset_parameters(module)

    input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
    block_type = "linear" if hyper_config.preprocess_beta else self.cfg.block_type
    self.beta_block = get_block(block_type)(
      input_dim, self.out_features + 1,
      hyper_config.include_sigmoid_activation)
    if self.cfg.include_output_layer:
      self.output_layer = nn.Linear(self.out_features,
                                    self.out_features,
                                    bias=False)
    else:
      self.output_layer = None

    if hyper_config.chunked:
      self.chunk_weight = torch.nn.Parameter(torch.empty((self.out_features, self.in_features)))
      if module.bias is not None:
        self.chunk_bias = torch.nn.Parameter(torch.empty(self.out_features))
      else:
        self.register_parameter("chunk_bias", None)
      self.reset_chunk_parameters(module)

      self.chunk_beta_block = get_block(block_type)(
        input_dim, self.out_features + 1,
        hyper_config.include_sigmoid_activation)
      if self.cfg.include_output_layer:
        self.chunk_output_layer = nn.Linear(self.out_features,
                                            self.out_features,
                                            bias=False)
      else:
        self.output_layer = None

    if self.hyper_type == "add":
      self.hyper_weight = torch.nn.Parameter(
        torch.empty((self.out_features, self.in_features)))
      if module.bias is not None:
        self.hyper_bias = torch.nn.Parameter(
          torch.empty(self.out_features))
      else:
        self.register_parameter("hyper_bias", None)
    self.reset_hyper_parameters()

  def reset_parameters(self, module):
    self.weight.data.copy_(module.weight.data)
    if module.bias is not None:
      self.bias.data.copy_(module.bias.data)

  def reset_chunk_parameters(self, module):
    self.chunk_weight.data.copy_(module.weight.data)
    if module.bias is not None:
      self.chunk_bias.data.copy_(module.bias.data)

  def reset_hyper_parameters(self):
    if self.output_layer is not None:
      init.eye_(self.output_layer.weight.data)
    if self.hyper_type == "add":
      init.kaiming_uniform_(self.hyper_weight, a=math.sqrt(5))
      if self.hyper_bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(
          self.hyper_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.hyper_bias, -bound, bound)

  def forward(self, inputs):
    hyper_out = self.beta_block(self._beta["net_beta"])
    # hyper_out = hyper_out * (self._beta["beta"] < 1.).int().float()
    # chunk_hyper_out = chunk_hyper_out * (self._beta["beta"] >= 1.).int().float()
    hyper_weight = hyper_out[:, :-1]
    hyper_bias = hyper_out[:, -1].unsqueeze(-1)

    if self.cfg.chunked:
      chunk_hyper_out = self.chunk_beta_block(self._beta["net_beta"])
      chunk_hyper_weight = chunk_hyper_out[:, :-1]
      chunk_hyper_bias = chunk_hyper_out[:, -1].unsqueeze(-1)

    if self.hyper_type == "add":
      out = F.linear(inputs, self.weight, self.bias)
      hyper_out = F.linear(inputs, self.hyper_weight) * hyper_weight
      if self.cfg.include_output_layer:
        hyper_out = self.output_layer(hyper_out)
      if self.hyper_bias is not None:
        hyper_out = hyper_out + self.hyper_bias.repeat(
          inputs.shape[0], 1) * hyper_bias
      out = out + hyper_out

    elif self.hyper_type == "s_add":
      out = F.linear(inputs, self.weight, self.bias)
      hyper_out = F.linear(inputs, self.weight) * hyper_weight
      if self.cfg.include_output_layer:
        hyper_out = self.output_layer(hyper_out)
      if self.bias is not None:
        hyper_out = hyper_out + self.bias.repeat(inputs.shape[0],
                                                 1) * hyper_bias
      out = out + hyper_out

    elif self.hyper_type == "ss_add":
      out = F.linear(inputs, self.weight)
      hyper_out = out + F.linear(inputs, self.weight) * hyper_weight
      if self.cfg.include_output_layer:
        hyper_out = self.output_layer(hyper_out)
      if self.bias is not None:
        hyper_out = hyper_out + self.bias
        hyper_out = hyper_out + self.bias.repeat(inputs.shape[0],
                                                 1) * hyper_bias
      out = out + hyper_out

      if self.cfg.chunked:
        chunk_out = F.linear(inputs, self.chunk_weight)
        chunk_hyper_out = chunk_out + F.linear(inputs, self.chunk_weight) * chunk_hyper_weight
        if self.cfg.include_output_layer:
          chunk_hyper_out = self.chunk_output_layer(chunk_hyper_out)
        if self.bias is not None:
          chunk_hyper_out = chunk_hyper_out + self.chunk_bias
          chunk_hyper_out = chunk_hyper_out + self.chunk_bias.repeat(inputs.shape[0],
                                                   1) * chunk_hyper_bias
        chunk_out = chunk_out + chunk_hyper_out

        out = out * (self._beta["beta"] < 1.).int().float() + chunk_out * (self._beta["beta"] >= 1.).int().float()

    elif self.hyper_type == "mult":
      out = F.linear(inputs, self.weight)
      out = out * hyper_weight
      if self.cfg.include_output_layer:
        out = self.output_layer(out)
      if self.bias is not None:
        out = out + self.bias.repeat(inputs.shape[0], 1) * hyper_bias
    else:
      raise ValueError

    return out
