import math
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.config import HyperConfig
from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.base_model import HyperVAE
from src.models.vq_vae import Quantizer

# Some constants used for sampling.
_SQRT3 = math.sqrt(3)
_LOG_RED_A = math.log(1)
_LOG_B = math.log(10)
_LOG_RED_M = (_LOG_RED_A + _LOG_B) / 2
_LOG_RED_DIFF = _LOG_RED_M - _LOG_RED_A


class HyperVQVAE(HyperVAE):

  def __init__(
      self,
      model_config: dict,
      hyper_cfg: HyperConfig,
      encoder: BaseHyperEncoder,
      decoder: BaseHyperDecoder,
  ) -> None:
    super().__init__(encoder=encoder, decoder=decoder, hyper_cfg=hyper_cfg)

    self.model_config = model_config
    self.model_name = "HyperVQVAE"

    # Dummy sample_dict for initialization.
    sample_dict = self.sample_inverse(torch.Tensor((1, 1)), 1)
    self.set_net_inputs(sample_dict["net"])
    self._set_quantizer(model_config)

  def sample(self, x: torch.Tensor) -> dict:
    device = x.device
    net = (torch.FloatTensor(1, 1).uniform_(-_SQRT3, _SQRT3).to(device))
    beta = net * (_SQRT3 / 3)
    beta = beta * _LOG_RED_DIFF + _LOG_RED_M
    sample_dict = {"net": net, "lamb": torch.exp(beta)}
    return sample_dict

  def sample_inverse(self, x: torch.Tensor, value: float) -> dict:
    device = x.device
    sample_dict = {}
    ones = torch.ones(1, 1).to(device)
    beta = value * ones
    net_beta = (torch.log(sample_dict["beta"]) - _LOG_RED_M) / _LOG_RED_DIFF
    sample_dict["net"] = net_beta * (3 / _SQRT3)
    sample_dict = {"net": net_beta * (3 / _SQRT3), "lamb": beta}
    return sample_dict

  def _set_quantizer(self, model_config: dict) -> None:
    if model_config["input_dim"] is None:
      raise AttributeError(
          "No input dimension provided !"
          "'input_dim' parameter of VQVAEConfig instance must be set to 'data_shape' where "
          "the shape of the data is (C, H, W ..). Unable to set quantizer.")

    x = torch.randn((2,) + self.model_config["input_dim"])
    z = self.encoder(x)["embedding"]
    if len(z.shape) == 2:
      z = z.reshape(z.shape[0], 1, 1, -1)

    z = z.permute(0, 2, 3, 1)

    self.model_config["embedding_dim"] = z.shape[-1]
    self.quantizer = Quantizer(model_config=model_config)

  def forward(self, inputs: dict, **kwargs) -> dict:
    # Default behaviour is to sample betas.
    return self.sample_forward(inputs)

  def sample_forward(self, inputs: dict, **kwargs) -> dict:
    x = inputs["data"]
    sample_dict = self.sample(x)
    self.set_net_inputs(sample_dict["net"])

    encoder_output = self.encoder(x)
    embeddings = encoder_output["embedding"]
    reshape_for_decoding = False

    if len(embeddings.shape) == 2:
      embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
      reshape_for_decoding = True

    embeddings = embeddings.permute(0, 2, 3, 1)

    quantizer_output = self.quantizer(embeddings)

    quantized_embed = quantizer_output["quantized_vector"]
    quantized_indices = quantizer_output["quantized_indices"]

    if reshape_for_decoding:
      quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)

    recon_x = self.decoder(quantized_embed)["reconstruction"]

    loss, recon_loss, vq_loss = self.loss_function(recon_x, x, quantizer_output, sample_dict["beta"])

    output = {
        "recon_loss": recon_loss,
        "vq_loss": vq_loss,
        "loss": loss,
        "recon_x": recon_x,
        "z": quantized_embed,
        "quantized_indices": quantized_indices,
    }

    return output

  def fixed_forward(self, inputs: dict, value: float, **kwargs) -> dict:
    x = inputs["data"]
    sample_dict = self.sample_inverse(x, value)
    self.set_net_inputs(sample_dict["net"])

    encoder_output = self.encoder(x)
    embeddings = encoder_output["embedding"]
    reshape_for_decoding = False

    if len(embeddings.shape) == 2:
      embeddings = embeddings.reshape(embeddings.shape[0], 1, 1, -1)
      reshape_for_decoding = True

    embeddings = embeddings.permute(0, 2, 3, 1)

    quantizer_output = self.quantizer(embeddings)

    quantized_embed = quantizer_output["quantized_vector"]
    quantized_indices = quantizer_output["quantized_indices"]

    if reshape_for_decoding:
      quantized_embed = quantized_embed.reshape(embeddings.shape[0], -1)

    recon_x = self.decoder(quantized_embed)["reconstruction"]

    loss, recon_loss, vq_loss = self.loss_function(recon_x, x, quantizer_output, None)

    output = {
        "recon_loss": recon_loss,
        "vq_loss": vq_loss,
        "loss": loss,
        "recon_x": recon_x,
        "z": quantized_embed,
        "quantized_indices": quantized_indices,
    }

    return output

  def loss_function(
      self,
      recon_x: torch.Tensor,
      x: torch.Tensor,
      quantizer_output: dict,
      lambdas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none").sum(dim=-1)

    vq_loss = quantizer_output["loss"]

    return (
        (recon_loss +
         lambdas.sum() * vq_loss).mean(dim=0) if lambdas is not None else
        (recon_loss + 1. * vq_loss).mean(dim=0),
        recon_loss.mean(dim=0),
        vq_loss.mean(dim=0),
    )

  def get_log_uniform_samples(self, num: int = 20) -> np.ndarray:
    return np.logspace(0, 1, num=num, base=10)


class HyperQuantizer(nn.Module):

  def __init__(self, model_config, hyper_cfg):
    nn.Module.__init__(self)

    self.model_config = model_config
    self.hyper_cfg = hyper_cfg

    self.embedding_dim = model_config["embedding_dim"]
    self.num_embeddings = model_config["num_embeddings"]
    self.commitment_loss_factor = model_config["commitment_loss_factor"]
    self.quantization_loss_factor = model_config["quantization_loss_factor"]

    self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
    self.embeddings.weight.data.uniform_(-1 / self.num_embeddings,
                                         1 / self.num_embeddings)

  def forward(self, z: torch.Tensor):
    distances = (
        (z.reshape(-1, self.embedding_dim)**2).sum(dim=-1, keepdim=True) +
        (self.embeddings.weight**2).sum(dim=-1) -
        2 * z.reshape(-1, self.embedding_dim) @ self.embeddings.weight.T)

    closest = distances.argmin(-1).unsqueeze(-1)

    quantized_indices = closest.reshape(z.shape[0], z.shape[1], z.shape[2])

    one_hot_encoding = (
        F.one_hot(closest,
                  num_classes=self.num_embeddings).type(torch.float).squeeze(1))

    # quantization
    quantized = one_hot_encoding @ self.embeddings.weight
    quantized = quantized.reshape_as(z)

    commitment_loss = F.mse_loss(
        quantized.detach().reshape(-1, self.embedding_dim),
        z.reshape(-1, self.embedding_dim),
        reduction="mean",
    )

    embedding_loss = F.mse_loss(
        quantized.reshape(-1, self.embedding_dim),
        z.detach().reshape(-1, self.embedding_dim),
        reduction="mean",
    ).mean(dim=-1)

    quantized = z + (quantized - z).detach()

    loss = (
        commitment_loss * self.commitment_loss_factor +
        embedding_loss * self.quantization_loss_factor)
    quantized = quantized.permute(0, 3, 1, 2)

    # output = ModelOutput(
    #   quantized_vector=quantized,
    #   quantized_indices=quantized_indices.unsqueeze(1),
    #   loss=loss,
    # )
    output = {
        "quantized_vector": quantized,
        "quantized_indices": quantized_indices.unsqueeze(1),
        "loss": loss,
    }

    return output
