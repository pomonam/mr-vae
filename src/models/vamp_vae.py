import torch
from torch import nn

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.base_model import VAE
import torch.nn.functional as F


class VampVAE(VAE):

  def __init__(
      self,
      encoder: BaseEncoder,
      decoder: BaseDecoder,
  ) -> None:
    super().__init__(encoder, decoder)

    self.model_name = "VampVAE"

    number_components = 50
    self.number_components = number_components
    input_dim = 784
    linear_layer = nn.Linear(number_components, input_dim)
    self.pseudo_inputs = nn.Sequential(linear_layer, nn.Hardtanh(0.0, 1.0))
    self.idle_input = torch.eye(number_components, requires_grad=False)

  def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
    if isinstance(inputs, torch.Tensor):
      x = inputs
    else:
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

  def loss_function(self, recon_x, x, mu, log_var, z, beta):
    recon_loss = F.binary_cross_entropy(
      recon_x.reshape(x.shape[0], -1),
      x.reshape(x.shape[0], -1),
      reduction="none",
    ).sum(dim=-1)

    log_p_z = self._log_p_z(z)

    log_q_z = (-0.5 * (log_var + torch.pow(z - mu, 2) / log_var.exp())).sum(dim=1)
    KLD = -(log_p_z - log_q_z)

    return (
      (recon_loss + beta * KLD).mean(dim=0),
      recon_loss.mean(dim=0),
      KLD.mean(dim=0),
    )
