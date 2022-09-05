import torch
from torch import nn

from src.config import HyperConfig
from src.hyper.layers import HyperLayer
from src.hyper.layers import initialize_hyper_blocks


def get_hyper_bn_layer(features: int, hyper_cfg: HyperConfig) -> HyperLayer:
  return HyperBatchNormLayer(features, hyper_cfg)


def get_hyper_ln_layer(features: int,
                       hyper_cfg: HyperConfig,
                       eps: float = 1e-12) -> HyperLayer:
  return HyperLayerNormLayer(features, hyper_cfg, eps)


class HyperBatchNormLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg

    if self.hyper_cfg.apply_norm_layers:
      if self.hyper_cfg.apply_bn_tracking:
        self.bn = nn.BatchNorm2d(
            features, affine=False, track_running_stats=True)
      else:
        self.bn = nn.BatchNorm2d(
            features, affine=False, track_running_stats=False)

      self.hyper_block_scale = initialize_hyper_blocks(self.features,
                                                       self.hyper_cfg)
      self.hyper_block_shift = initialize_hyper_blocks(self.features,
                                                       self.hyper_cfg)
    else:
      self.bn = nn.BatchNorm2d(features)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = self.bn(inputs)

    if not self.hyper_cfg.apply_norm_layers:
      return inputs
    else:
      scale = self.hyper_block_scale(self._net_inputs)
      shift = self.hyper_block_shift(self._net_inputs)

      # This is currently only applied to conv layers.
      scale = scale.unsqueeze(-1).unsqueeze(-1)
      shift = shift.unsqueeze(-1).unsqueeze(-1)

      # Initialized to be identity.
      return (1 + scale) * inputs + shift


class HyperLayerNormLayer(HyperLayer):

  def __init__(self,
               features: int,
               hyper_cfg: HyperConfig,
               eps: float = 1e-12) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg

    if self.hyper_cfg.apply_norm_layers:
      self.ln = nn.LayerNorm(features, elementwise_affine=False, eps=eps)

      self.hyper_block_scale = initialize_hyper_blocks(self.features,
                                                       self.hyper_cfg)
      self.hyper_block_shift = initialize_hyper_blocks(self.features,
                                                       self.hyper_cfg)
    else:
      self.ln = nn.LayerNorm(features, eps=eps)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = self.ln(inputs)

    if not self.hyper_cfg.apply_norm_layers:
      return inputs
    else:
      scale = self.hyper_block_scale(self._net_inputs)
      shift = self.hyper_block_shift(self._net_inputs)

      # This is currently only applied to conv layers.
      scale = scale.unsqueeze(-1).unsqueeze(-1)
      shift = shift.unsqueeze(-1).unsqueeze(-1)

      # Initialized to be identity.
      return (1 + scale) * inputs + shift
