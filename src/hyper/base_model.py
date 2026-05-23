"""Hyper-network VAE base class.

Implements log-uniform β sampling (Algorithm 1) and input standardization
(Section 3.4) from the paper. Each layer of the base VAE has a per-layer gate
(HyperSigmoidLayer for encoders, HyperSqrtLayer for decoders); this class
broadcasts the hyper-input to every gate before the forward pass.
"""
import math

import numpy as np
import torch

from src.base_model import VAE
from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder

_SQRT3 = math.sqrt(3.0)


class HyperVAE(VAE):

  def __init__(
      self,
      encoder: BaseHyperEncoder,
      decoder: BaseHyperDecoder,
      sample_a: float = 0.01,
      sample_b: float = 10.0,
  ) -> None:
    super().__init__(encoder=encoder, decoder=decoder)
    self.model_name = "HyperVAE"
    self.sample_a = sample_a
    self.sample_b = sample_b
    self._log_a = math.log(sample_a)
    self._log_b = math.log(sample_b)
    self._log_m = 0.5 * (self._log_a + self._log_b)
    self._log_half_range = 0.5 * (self._log_b - self._log_a)

  def set_net_inputs(self, net_inputs: torch.Tensor) -> None:
    self.encoder.set_net_inputs(net_inputs)
    self.decoder.set_net_inputs(net_inputs)

  def forward(self, inputs, **kwargs):
    raise NotImplementedError()

  def sample(self, x: torch.Tensor) -> dict:
    """Draw β ~ log-Uniform[a, b] for each example, and return both the raw β
    and the standardized hyper-network input.
    """
    batch_size = x.shape[0]
    device = x.device
    net = torch.empty(batch_size, 1, device=device).uniform_(-_SQRT3, _SQRT3)
    log_beta = (net / _SQRT3) * self._log_half_range + self._log_m
    return {"net": net, "beta": torch.exp(log_beta)}

  def sample_inverse(self, x: torch.Tensor, value: float) -> dict:
    """Inverse of `sample`: given a target β, produce the matching hyper-network input."""
    batch_size = x.shape[0]
    device = x.device
    beta = torch.full((batch_size, 1), value, device=device)
    net = (torch.log(beta) - self._log_m) / self._log_half_range * _SQRT3
    return {"net": net, "beta": beta}

  def get_log_uniform_samples(self, num: int = 20) -> np.ndarray:
    """Return `num` β values log-uniformly spaced in [sample_a, sample_b] for evaluation."""
    return np.logspace(self._log_a / math.log(10),
                       self._log_b / math.log(10),
                       num=num,
                       base=10)
