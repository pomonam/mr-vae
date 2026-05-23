"""Per-layer pre-activation modulation modules (Eqns 6-8 and Listing 1 of the paper)."""
from typing import Optional

import torch
from torch import nn


class HyperLayer(nn.Module):
  """Base class. Stores the hypernetwork input (standardized log β) set by the
  parent HyperVAE before each forward pass.
  """
  _net_inputs: Optional[torch.Tensor] = None

  def set_net_inputs(self, value: torch.Tensor) -> None:
    self._net_inputs = value

  def reset_net_inputs(self) -> None:
    self._net_inputs = None


def get_hyper_layer(features: int, decoder: bool = False) -> HyperLayer:
  return HyperSqrtLayer(features) if decoder else HyperSigmoidLayer(features)


class HyperSigmoidLayer(HyperLayer):
  """Encoder gate: σ(W · log(β) + b) ⊙ x, with σ = sigmoid (paper Eqn 7)."""

  def __init__(self, features: int) -> None:
    super().__init__()
    self.features = features
    self.hyper_block_scale = nn.Linear(1, features, bias=True)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    scale = torch.sigmoid(self.hyper_block_scale(self._net_inputs))
    if inputs.dim() == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)
    return scale * inputs


class HyperSqrtLayer(HyperLayer):
  """Decoder gate: σ(W · log(β) + b) ⊙ x, with σ(x) = √(ReLU(1 − exp(x))) (paper Eqn 7)."""

  def __init__(self, features: int) -> None:
    super().__init__()
    self.features = features
    self.hyper_block_scale = nn.Linear(1, features, bias=True)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    raw = self.hyper_block_scale(self._net_inputs)
    scale = torch.sqrt(torch.relu(1 - torch.exp(raw)))
    if inputs.dim() == 4:
      scale = scale.unsqueeze(-1).unsqueeze(-1)
    return scale * inputs
