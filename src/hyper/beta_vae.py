import math

import torch

from src.config import HyperConfig
from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.base_model import HyperVAE


class HyperBetaVAE(HyperVAE):

  def __init__(self,
               hyper_cfg: HyperConfig,
               encoder: BaseHyperEncoder,
               decoder: BaseHyperDecoder) -> None:
    super().__init__(encoder=encoder, decoder=decoder, hyper_cfg=hyper_cfg)
    self.encoder = encoder
    self.decoder = decoder
    self.model_name = "BetaVAE"

  def forward(self, inputs: dict, **kwargs) -> dict:
    # Default behaviour is to sample betas.
    return self.sample_forward(inputs)

  def sample_forward(self, inputs: dict, **kwargs) -> dict:
    x = inputs["data"]
    sample_dict = self.sample(x)
    self.set_net_inputs(sample_dict["net"])

    encoder_output = self.encoder(x)
    mu, log_var = encoder_output["embedding"], encoder_output["log_covariance"]
    std = torch.exp(0.5 * log_var)
    z, eps = self._sample_gauss(mu, std)
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
        "beta": sample_dict["beta"]
    }

    return output

  def fixed_forward(self, inputs: dict, value: float, **kwargs) -> dict:
    x = inputs["data"]
    sample_dict = self.sample_inverse(x, value)
    self.set_net_inputs(sample_dict["net"])

    encoder_output = self.encoder(x)
    mu, log_var = encoder_output["embedding"], encoder_output["log_covariance"]
    std = torch.exp(0.5 * log_var)
    z, eps = self._sample_gauss(mu, std)
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
        "beta": sample_dict["beta"]
    }
    return output
