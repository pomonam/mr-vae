import math

import numpy as np
import torch

from src.base_model import VAE
from src.config import HyperConfig
from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.blocks import get_block

# Some constants used for sampling.
_SQRT3 = math.sqrt(3)
_LOG_A = math.log(0.001)
# If you want the samping to start from 0.01, uncomment below.
# _LOG_A = math.log(0.01)
_LOG_B = math.log(10)
_LOG_M = (_LOG_A + _LOG_B) / 2
_LOG_DIFF = (_LOG_M - _LOG_A)


class HyperVAE(VAE):

  def __init__(self,
               hyper_cfg: HyperConfig,
               encoder: BaseHyperEncoder,
               decoder: BaseHyperDecoder,
               reconstruction_loss: str = "mse") -> None:
    super().__init__(
        encoder=encoder,
        decoder=decoder,
        reconstruction_loss=reconstruction_loss)

    self.model_name = "HyperVAE"
    self.hyper_cfg = hyper_cfg

    if self.hyper_cfg.shared_preprocess:
      self.preprocess_block = get_block(self.hyper_cfg.block_type)(
          in_features=1,
          out_features=self.hyper_cfg.shared_preprocess_dim,
          # By default, hidden dimension has expansion factor of 4.
          emd_features=self.hyper_cfg.shared_preprocess_dim * 4)

  def set_net_inputs(self, value: torch.Tensor) -> None:
    if self.hyper_cfg.shared_preprocess:
      value = self.preprocess_block(value)
    self.encoder.set_net_inputs(value)
    self.decoder.set_net_inputs(value)

  def forward(self, inputs: torch.Tensor, **kwargs):
    raise NotImplementedError

  # noinspection PyMethodMayBeStatic
  def sample(self, x: torch.Tensor) -> dict:
    try:
      batch_size = x.shape[0]
      device = x.device
    except AttributeError:
      # This is for text models.
      batch_size = x.batch_size
      device = x._batch["text_ids"].device
    sample_dict = {}
    sample_dict["net"] = \
      torch.FloatTensor(batch_size, 1).uniform_(-_SQRT3, _SQRT3).to(device)
    beta = sample_dict["net"] * (_SQRT3 / 3)
    beta = beta * _LOG_DIFF + _LOG_M
    sample_dict["beta"] = torch.exp(beta)
    return sample_dict

  # noinspection PyMethodMayBeStatic
  def sample_inverse(self, x: torch.Tensor, value: float) -> dict:
    try:
      batch_size = x.shape[0]
      device = x.device
    except AttributeError:
      # This is for text models.
      batch_size = x.batch_size
      device = x._batch["text_ids"].device
    sample_dict = {}
    ones = torch.ones(batch_size, 1).to(device)
    beta = value * ones
    sample_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta
    net_beta = (torch.log(sample_dict["beta"]) - _LOG_M) / _LOG_DIFF
    sample_dict["net"] = net_beta * (3 / _SQRT3)
    return sample_dict

  # noinspection PyMethodMayBeStatic
  def get_log_uniform_samples(self, num: int = 20) -> np.ndarray:
    # Log-uniform sampling between 0.01 and 10.
    # return np.logspace(-2, 1, num=num, base=10)
    return np.logspace(-3, 1, num=num, base=10)
