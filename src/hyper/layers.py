import torch
import torch.nn as nn

from src.config import HyperConfig
from src.hyper.blocks import get_block


def get_hyper_layer(features, hyper_cfg):
  _dict = {
    "sig_gate": HyperSigmoidLayer(features, hyper_cfg),
    "tanh_gate": HyperTanhLayer(features, hyper_cfg)
  }
  return _dict[hyper_cfg.param_type]


class HyperLayer(nn.Module):
  # _net_inputs = None

  def set_net_inputs(self, value: torch.Tensor) -> None:
    self._net_inputs = value

  def reset_net_inputs(self) -> None:
    self._net_inputs = None


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
