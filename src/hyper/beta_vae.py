import torch

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.base_model import HyperVAE


class HyperBetaVAE(HyperVAE):

  def __init__(
      self,
      encoder: BaseHyperEncoder,
      decoder: BaseHyperDecoder,
      sample_a: float = 0.01,
      sample_b: float = 10.0,
  ) -> None:
    super().__init__(
        encoder=encoder,
        decoder=decoder,
        sample_a=sample_a,
        sample_b=sample_b)
    self.model_name = "HyperBetaVAE"

  def forward(self, inputs, **kwargs) -> dict:
    return self.sample_forward(inputs)

  def _encode_decode(self, x: torch.Tensor) -> dict:
    encoder_output = self.encoder(x)
    mu = encoder_output["embedding"]
    log_var = encoder_output["log_covariance"]
    std = torch.exp(0.5 * log_var)
    z, _ = self._sample_gauss(mu, std)
    recon_x = self.decoder(z)["reconstruction"]
    return {"reconstruction": recon_x, "mu": mu, "log_var": log_var, "z": z}

  def sample_forward(self, inputs, **kwargs) -> dict:
    x = inputs if isinstance(inputs, torch.Tensor) else inputs["data"]
    sample_dict = self.sample(x)
    self.set_net_inputs(sample_dict["net"])
    out = self._encode_decode(x)
    out["data"] = x
    out["beta"] = sample_dict["beta"]
    return out

  def fixed_forward(self, inputs, value: float, **kwargs) -> dict:
    x = inputs if isinstance(inputs, torch.Tensor) else inputs["data"]
    sample_dict = self.sample_inverse(x, value)
    self.set_net_inputs(sample_dict["net"])
    out = self._encode_decode(x)
    out["data"] = x
    out["beta"] = sample_dict["beta"]
    return out
