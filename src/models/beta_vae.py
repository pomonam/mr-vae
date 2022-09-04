import torch

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.base_model import VAE


class BetaVAE(VAE):

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
