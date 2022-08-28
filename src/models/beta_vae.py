import torch

from src.base_model import VAE
import math


def log_sum_exp(value, dim=None, keepdim=False):
  if dim is not None:
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
      m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
  else:
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)


class BetaVAE(VAE):

  def __init__(
      self,
      encoder=None,
      decoder=None,
  ):
    super().__init__()
    # BetaVAE.__init__(self, encoder=encoder, decoder=decoder)
    self.encoder = encoder
    self.decoder = decoder
    self.model_name = "BetaVAE"

  def forward(self, inputs, **kwargs):
    x = inputs["data"]

    encoder_output = self.encoder(x)

    mu, log_var = encoder_output["embedding"], encoder_output["log_covariance"]

    std = torch.exp(0.5 * log_var)
    z, eps = self._sample_gauss(mu, std)
    try:
      recon_x = self.decoder(z)["reconstruction"]
    except LookupError:
      recon_x = self.decoder.ar_forward(x, z)

    output = {
        "reconstruction": recon_x,
        "data": x,
        "mu": mu,
        "log_var": log_var,
        "z": z,
    }

    return output

  def eval_forward(self, inputs, **kwargs):
    x = inputs["data"]

    encoder_output = self.encoder(x)

    mu, log_var = encoder_output["embedding"], encoder_output["log_covariance"]

    std = torch.exp(0.5 * log_var)
    z, eps = self._sample_gauss(mu, std)
    try:
      recon_x = self.decoder(z)["reconstruction"]
    except LookupError:
      recon_x = self.decoder.ar_forward(x, z)

    batch_size, nz = mu.size()
    neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 *
                   (1 + log_var).sum(-1)).mean()
    var = log_var.exp()
    dev = z - mu
    log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                  0.5 * (nz * math.log(2 * math.pi) + log_var.sum(-1))
    log_qz = log_sum_exp(log_density, dim=1) - math.log(batch_size)
    mi = (neg_entropy - log_qz.mean(-1)).item()

    output = {
        "reconstruction": recon_x,
        "data": x,
        "mu": mu,
        "log_var": log_var,
        "z": z,
        "mi": mi
    }

    return output

  def _sample_gauss(self, mu, std):
    eps = torch.randn_like(std)
    return mu + eps * std, eps
