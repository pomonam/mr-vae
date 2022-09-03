import torch
import torch.nn as nn

from src.config import HyperConfig
from src.hyper.blocks import get_block
import torch.nn.functional as F

def get_hyper_layer(features, hyper_cfg):
  _dict = {
    "sig_gate": HyperSigmoidLayer(features, hyper_cfg),
    "tanh_gate": HyperTanhLayer(features, hyper_cfg),
    "scale_shift": HyperScaleShiftLayer(features, hyper_cfg)
  }
  return _dict[hyper_cfg.layer_type]


def get_hyper_bn_layer(features, hyper_cfg):
  # return nn.BatchNorm2d(features)
  return HyperBatchNormLayer(features, hyper_cfg)


class HyperLayer(nn.Module):
  # _net_inputs = None

  def set_net_inputs(self, value: torch.Tensor) -> None:
    self._net_inputs = value

  def reset_net_inputs(self) -> None:
    self._net_inputs = None


class HyperBatchNormLayer(HyperLayer):

  def __init__(self, features, hyper_cfg):
    super().__init__()

    self._net_inputs = None
    self.features = features
    self.hyper_cfg = hyper_cfg

    self.bn = nn.BatchNorm2d(features, affine=False, track_running_stats=False)

    if hyper_cfg.preprocess_beta:
      self.hyper_block_scale = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
      self.hyper_block_shift = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
    else:
      self.hyper_block_scale = get_block(self.hyper_cfg.block_type)(
        in_features=1, width=self.features)
      self.hyper_block_shift = get_block(self.hyper_cfg.block_type)(
        in_features=1, width=self.features)

  def forward(self, inputs):
    inputs = self.bn(inputs)

    scale = self.hyper_block_scale(self._net_inputs)
    shift = self.hyper_block_shift(self._net_inputs)

    scale = scale.unsqueeze(-1).unsqueeze(-1)
    shift = shift.unsqueeze(-1).unsqueeze(-1)
    return scale * inputs + shift

  # def forward(self, inputs):
  #   batch_size, channels, height, width = inputs.data.shape
  #
  #   scale = self.hyper_block_scale(self._net_inputs)
  #   scale = torch.sigmoid(scale)
  #
  #   # Get the mean and variance
  #   batch_mean = torch.mean(inputs)
  #   batch_var = torch.var(inputs)
  #
  #   if len(inputs.shape) == 4:
  #     scale = scale.unsqueeze(-1).unsqueeze(-1)
  #
  #   return scale * inputs

class HyperSigmoidLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig):
    super().__init__()

    self._net_inputs = None
    self.features = features
    self.hyper_cfg = hyper_cfg

    if hyper_cfg.preprocess_beta:
      self.hyper_block_scale = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
    else:
      self.hyper_block_scale = get_block(self.hyper_cfg.block_type)(
        in_features=1, width=self.hyper_cfg.preprocess_dim)

  def forward(self, inputs):
    scale = self.hyper_block_scale(self._net_inputs)
    scale = torch.sigmoid(scale)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)

    return scale * inputs


class HyperTanhLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig):
    super().__init__()

    self._net_inputs = None
    self.features = features
    self.hyper_cfg = hyper_cfg

    if hyper_cfg.preprocess_beta:
      self.hyper_block_scale = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
    else:
      self.hyper_block_scale = get_block(self.hyper_cfg.block_type)(
        in_features=1, width=self.features)

  def forward(self, inputs):
    scale = self.hyper_block_scale(self._net_inputs)
    scale = torch.tanh(scale)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)

    return inputs + scale * inputs


class HyperScaleShiftLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig):
    super().__init__()

    self._net_inputs = None
    self.features = features
    self.hyper_cfg = hyper_cfg

    if hyper_cfg.preprocess_beta:
      self.hyper_block_scale = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
      self.hyper_block_shift = get_block("linear")(hyper_cfg.preprocess_dim, self.features)
    else:
      self.hyper_block_scale = get_block(self.hyper_cfg.block_type)(
        in_features=1, width=self.features)
      self.hyper_block_shift = get_block(self.hyper_cfg.block_type)(
        in_features=1, width=self.features)

  def forward(self, inputs):
    scale = self.hyper_block_scale(self._net_inputs)
    shift = self.hyper_block_shift(self._net_inputs)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)
      shift = shift.unsqueeze(-1).unsqueeze(-1)

    return scale * inputs + shift
