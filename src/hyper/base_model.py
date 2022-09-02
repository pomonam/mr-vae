import math

import numpy as np
import torch

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.base_model import VAE
from src.config import HyperConfig
from src.hyper.blocks import get_block

_SQRT3 = math.sqrt(3)
_LOG_A = math.log(0.001)
_LOG_B = math.log(10)
_LOG_M = (_LOG_A + _LOG_B) / 2
_LOG_DIFF = (_LOG_M - _LOG_A)


class HyperVAE(VAE):

  def __init__(self,
               hyper_cfg: HyperConfig,
               encoder: BaseHyperEncoder = None,
               decoder: BaseHyperDecoder = None,
               reconstruction_loss: str = "mse"):
    VAE.__init__(
        self,
        encoder=encoder,
        decoder=decoder,
        reconstruction_loss=reconstruction_loss)

    self.model_name = "VAE"
    self.hyper_cfg = hyper_cfg

    if self.hyper_cfg.preprocess_beta:
      self.preprocess_block = get_block(self.hyper_cfg.block_type)(
          in_features=1, width=self.hyper_cfg.preprocess_dim)

  def set_net_inputs(self, value: torch.Tensor) -> None:
    if self.hyper_cfg.preprocess_beta:
      value = self.preprocess_block(value)
    self.encoder.set_net_inputs(value)
    try:
      self.decoder.set_net_inputs(value)
    except:
      pass

  def forward(self, x, **kwargs):
    pass

  def sample(self, x: torch.Tensor):
    batch_size = x.shape[0]
    device = x.device
    sample_dict = dict()
    sample_dict["net"] = \
      torch.FloatTensor(batch_size, 1).uniform_(-_SQRT3, _SQRT3).to(device)
    beta = sample_dict["net"] * (_SQRT3 / 3)
    beta = beta * _LOG_DIFF + _LOG_M
    sample_dict["beta"] = torch.exp(beta)
    return sample_dict

  def sample_inverse(self, x: torch.Tensor, value: float):
    batch_size = x.shape[0]
    device = x.device
    sample_dict = dict()
    ones = torch.ones(batch_size, 1).to(device)
    beta = value * ones
    sample_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta
    net_beta = (torch.log(sample_dict["beta"]) - _LOG_M) / _LOG_DIFF
    sample_dict["net"] = net_beta * (3 / _SQRT3)
    return sample_dict

  def get_test_samples(self, num=20):
    return np.logspace(-3, 1, num=num, base=10)
