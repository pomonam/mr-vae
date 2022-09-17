from typing import Optional

import torch
from torch import nn

from src.config import HyperConfig
from src.hyper.blocks import get_block


class HyperLayer(nn.Module):
  _net_inputs: Optional[torch.Tensor] = None

  def set_net_inputs(self, value: torch.Tensor) -> None:
    self._net_inputs = value

  def reset_net_inputs(self) -> None:
    self._net_inputs = None


def get_hyper_layer(features: int,
                    hyper_cfg: HyperConfig,
                    decoder: bool = False) -> HyperLayer:
  assert decoder is not None

  hyper_dict = {
      "sig_gate": HyperSigmoidLayer(features, hyper_cfg),
      "sqrt_gate": HyperSqrtLayer(features, hyper_cfg),
      "tanh_gate": HyperTanhLayer(features, hyper_cfg),
      "scale_shift": HyperScaleShiftLayer(features, hyper_cfg),
      "affine": HyperAffineLayer(features, hyper_cfg)
  }
  if decoder:
    return hyper_dict[hyper_cfg.decoder_layer_type]
  else:
    # For encoders ...
    return hyper_dict[hyper_cfg.encoder_layer_type]


def initialize_hyper_blocks(features: int,
                            hyper_cfg: HyperConfig,
                            force_apply_zero_init: bool = False) -> nn.Module:
  if hyper_cfg.shared_preprocess:
    block = nn.Linear(hyper_cfg.shared_preprocess_dim, features, bias=True)
    if hyper_cfg.apply_zero_init or force_apply_zero_init:
      block.weight.data.fill_(0)
      block.bias.data.fill_(0)
  else:
    block = get_block(hyper_cfg.block_type)(
        # Scalar inputs for non-shared preprocess.
        in_features=1,
        out_features=features,
        emd_features=hyper_cfg.non_shared_emd_dim,
    )
    if hyper_cfg.apply_zero_init or force_apply_zero_init:
      block.layers[-1].weight.data.fill_(0)
      block.layers[-1].bias.data.fill_(0)
  return block


class HyperSigmoidLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg
    self.hyper_block_scale = \
      initialize_hyper_blocks(self.features, self.hyper_cfg)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    scale = self.hyper_block_scale(self._net_inputs)
    scale = torch.sigmoid(scale)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)

    if len(inputs.shape) == 3:
      scale = scale.unsqueeze(1)

    return scale * inputs


class HyperSqrtLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg
    self.hyper_block_scale = \
      initialize_hyper_blocks(self.features, self.hyper_cfg)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    scale = self.hyper_block_scale(self._net_inputs)
    scale = torch.relu(1 - torch.exp(scale))
    # Inputs should be >= 0.
    scale = torch.sqrt(scale)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)

    if len(inputs.shape) == 3:
      scale = scale.unsqueeze(1)

    return scale * inputs


class HyperAffineLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg
    self.hyper_block_scale = \
      initialize_hyper_blocks(self.features, self.hyper_cfg)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    scale = self.hyper_block_scale(self._net_inputs)
    scale = torch.relu(scale)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)

    if len(inputs.shape) == 3:
      scale = scale.unsqueeze(1)

    return scale * inputs


class HyperTanhLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg
    self.hyper_block_scale = \
      initialize_hyper_blocks(self.features, self.hyper_cfg)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    scale = self.hyper_block_scale(self._net_inputs)
    scale = torch.tanh(scale)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)

    if len(inputs.shape) == 3:
      scale = scale.unsqueeze(1)

    # Residual connection for tanh transformation.
    # Expression: I + tanh(x).
    return inputs + scale * inputs


class HyperScaleShiftLayer(HyperLayer):

  def __init__(self, features: int, hyper_cfg: HyperConfig) -> None:
    super().__init__()

    self.features = features
    self.hyper_cfg = hyper_cfg
    self.hyper_block_scale = \
      initialize_hyper_blocks(self.features, self.hyper_cfg)
    self.hyper_block_shift = \
      initialize_hyper_blocks(self.features, self.hyper_cfg)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    scale = self.hyper_block_scale(self._net_inputs)
    shift = self.hyper_block_shift(self._net_inputs)

    if len(inputs.shape) == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)
      shift = shift.unsqueeze(-1).unsqueeze(-1)

    if len(inputs.shape) == 3:
      scale = scale.unsqueeze(1)
      shift = shift.unsqueeze(1)

    # Can optionally do (1 + scale) * inputs + shift
    return scale * inputs + shift
