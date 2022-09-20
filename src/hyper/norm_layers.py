import torch
from torch import nn

from src.config import HyperConfig
from src.hyper.blocks import get_block
from src.hyper.layers import HyperLayer


def get_hyper_bn_layer(features: int, hyper_cfg: HyperConfig) -> HyperLayer:
  return HyperBatchNormLayer(features, hyper_cfg)


def calibrate_bn(module: nn.Module):
  # Reset BN batch statistics.
  if isinstance(module, nn.BatchNorm2d):
    # Reset all values.
    module.reset_running_stats()
    module.track_running_stats = True
    module.training = True
    # Using cumulative moving average.
    module.momentum = None


def get_hyper_ln_layer(features: int,
                       hyper_cfg: HyperConfig,
                       eps: float = 1e-12) -> HyperLayer:
  return HyperLayerNormLayer(features, hyper_cfg, eps)


def initialize_hyper_norm_blocks(
    features: int,
    hyper_cfg: HyperConfig,
    force_apply_zero_init: bool = False) -> nn.Module:
  # The default is MLP for BN layer.
  block = get_block("mlp")(
      in_features=1,
      out_features=features,
      emd_features=hyper_cfg.non_shared_emd_dim,
  )
  if hyper_cfg.apply_zero_init or force_apply_zero_init:
    block.layers[-1].weight.data.fill_(0)
    block.layers[-1].bias.data.fill_(0)
  return block


class HyperBatchNormLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg

    if self.hyper_cfg.norm_type == "scale_shift":
      # Default parameters:
      self.weight = nn.Parameter(torch.empty(features), requires_grad=True)
      self.bias = nn.Parameter(torch.empty(features), requires_grad=True)
      torch.nn.init.ones_(self.weight)
      torch.nn.init.zeros_(self.bias)

      if not self.hyper_cfg.apply_bn_replace:
        self.bn = nn.BatchNorm2d(
            features, affine=False, track_running_stats=True)
      else:
        # If replace flag is up, replace with InstanceNorm.
        self.bn = nn.InstanceNorm2d(features, affine=False)

      self.hyper_block_scale = initialize_hyper_norm_blocks(
          self.features, self.hyper_cfg)
      self.hyper_block_shift = initialize_hyper_norm_blocks(
          self.features, self.hyper_cfg)
    else:
      self.bn = nn.BatchNorm2d(features, affine=True, track_running_stats=True)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    try:
      inputs = self.bn(inputs)
    except ValueError:
      # InstanceNorm might not work here.
      pass

    if self.hyper_cfg.norm_type == "scale_shift":
      scale = self.hyper_block_scale(self._net_inputs)
      shift = self.hyper_block_shift(self._net_inputs)

      # This is currently only applied to conv layers.
      weight = self.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
      bias = self.bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
      scale = scale.unsqueeze(-1).unsqueeze(-1)
      shift = shift.unsqueeze(-1).unsqueeze(-1)

      return (weight + scale) * inputs + (bias + shift)

    else:
      return inputs


class HyperLayerNormLayer(HyperLayer):

  def __init__(self,
               features: int,
               hyper_cfg: HyperConfig,
               eps: float = 1e-12) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg

    if self.hyper_cfg.norm_type == "scale_shift":
      # Default parameters:
      self.weight = nn.Parameter(torch.empty(features))
      self.bias = nn.Parameter(torch.empty(features))
      torch.nn.init.ones_(self.weight)
      torch.nn.init.zeros_(self.bias)

      self.ln = nn.LayerNorm(features, elementwise_affine=False, eps=eps)
      self.hyper_block_scale = initialize_hyper_norm_blocks(
          self.features, self.hyper_cfg)
      self.hyper_block_shift = initialize_hyper_norm_blocks(
          self.features, self.hyper_cfg)
    else:
      self.ln = nn.LayerNorm(features, elementwise_affine=True, eps=eps)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    inputs = self.ln(inputs)

    if self.hyper_cfg.norm_type == "scale_shift":
      scale = self.hyper_block_scale(self._net_inputs).unsqueeze(1)
      shift = self.hyper_block_shift(self._net_inputs).unsqueeze(1)
      weight = self.weight.unsqueeze(0).unsqueeze(0)
      bias = self.bias.unsqueeze(0).unsqueeze(0)
      return (weight + scale) * inputs + (bias + shift)
    else:
      return inputs
