from typing import Tuple

import torch
from torch import nn

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder


class BaseAE(nn.Module):

  def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder) -> None:
    super().__init__()
    self.model_name = "BaseAE"
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, inputs: torch.Tensor, **kwargs):
    raise NotImplementedError()


class VAE(BaseAE):

  def __init__(self, encoder: BaseEncoder, decoder: BaseDecoder) -> None:
    super().__init__(encoder=encoder, decoder=decoder)
    self.model_name = "VAE"

  def forward(self, inputs: torch.Tensor, **kwargs):
    raise NotImplementedError()

  @staticmethod
  def _sample_gauss(mu: torch.Tensor,
                    std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.randn_like(std)
    return mu + eps * std, eps
