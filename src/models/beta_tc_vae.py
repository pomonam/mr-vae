import numpy as np
import torch

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.base_model import VAE


class BetaTCVAE(VAE):

  def __init__(
      self,
      encoder: BaseEncoder,
      decoder: BaseDecoder,
  ) -> None:
    super().__init__(encoder, decoder)

    self.model_name = "BetaVAE"

  def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
    x = inputs["data"]

    encoder_output = self.encoder(x)
    mu, log_var = encoder_output["embedding"], encoder_output["log_covariance"]
    std = torch.exp(0.5 * log_var)
    z, _ = self._sample_gauss(mu, std)
    try:
      recon_x = self.decoder(z)["reconstruction"]
    except LookupError:
      recon_x = self.decoder.ar_forward(x, z, inputs)

    output = {
        "reconstruction": recon_x,
        "data": x,
        "mu": mu,
        "log_var": log_var,
        "z": z,
    }

    return output

  @staticmethod
  def _compute_log_gauss_density(z, mu, log_var):
    """element-wise computation"""
    return -0.5 * (
        torch.log(torch.tensor([2 * np.pi]).to(z.device)) + log_var +
        (z - mu)**2 * torch.exp(-log_var))

  @staticmethod
  def _log_importance_weight_matrix(batch_size, dataset_size):
    """Compute importance weigth matrix for MSS
    Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
    """
    n = dataset_size
    m = batch_size - 1
    strat_weight = (n - m) / (n * m)
    w = torch.Tensor(batch_size, batch_size).fill_(1 / m)
    w.view(-1)[::m + 1] = 1 / n
    w.view(-1)[1::m + 1] = strat_weight
    w[m - 1, 0] = strat_weight
    return w.log()
