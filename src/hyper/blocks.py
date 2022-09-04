from typing import Optional

import torch
from torch import nn


class BaseBlock(nn.Module):

  def __init__(self,
               in_features: int,
               out_features: int,
               emd_features: Optional[int] = None):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.emd_features = out_features if emd_features is None else emd_features

    self.layers = None
    self._construct_layers()

  def _construct_layers(self) -> None:
    raise NotImplementedError

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    out = self.layers(inputs)
    return out


def get_block(name: str) -> BaseBlock:
  _BLOCK_DICT = {
      "linear": LinearBlock,
      "mlp": MlpBlock,
      "large_mlp": LargeMlpBlock,
  }
  return _BLOCK_DICT[name]


class LinearBlock(BaseBlock):

  def _construct_layers(self) -> None:
    self.layers = nn.Sequential(
        nn.Linear(self.in_features, self.out_features, bias=True),)


class MlpBlock(BaseBlock):

  def _construct_layers(self) -> None:
    self.layers = nn.Sequential(
        nn.Linear(self.in_features, self.emd_features),
        nn.ReLU(inplace=True),
        nn.Linear(self.emd_features, self.out_features),
    )


class LargeMlpBlock(BaseBlock):

  def _construct_layers(self) -> None:
    self.layers = nn.Sequential(
        nn.Linear(self.in_features, self.emd_features),
        nn.GELU(),
        nn.Linear(self.emd_features, self.emd_features),
        nn.GELU(),
        nn.Linear(self.emd_features, self.out_features),
    )
