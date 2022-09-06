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

    self.model_name = "BetaTCVAE"

  def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
    x = inputs["data"]

    encoder_output = self.encoder(x)
    # It is changed to log_std here.
    mu, log_std = encoder_output["embedding"], encoder_output["log_covariance"]
    std = torch.exp(log_std)
    z, _ = self._sample_gauss(mu, std)
    try:
      recon_x = self.decoder(z)["reconstruction"]
    except LookupError:
      recon_x = self.decoder.ar_forward(x, z, inputs)

    output = {
        "reconstruction": recon_x,
        "data": x,
        "mu": mu,
        # This is bad practice!
        "log_var": log_std,
        "z": z,
    }

    return output
