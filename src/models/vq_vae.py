# Code adapted from:
# https://github.com/clementchadebec/benchmark_VAE/blob/main/src/pythae/models/vq_vae/vq_vae_model.py
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.base_model import BaseAE


class VQVAE(BaseAE):

  def __init__(
      self,
      model_config: dict,
      encoder: BaseEncoder,
      decoder: BaseDecoder,
      lamb: float = 1.0,
  ) -> None:
    super().__init__(encoder=encoder, decoder=decoder)

    self.model_config = model_config
    self.model_name = "VQVAE"
    self.lamb = lamb
    self._set_quantizer(model_config)

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

  def forward(self, inputs: torch.Tensor, **kwargs) -> dict:
    if isinstance(inputs, torch.Tensor):
      x = inputs
    else:
      x = inputs["data"]

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

    loss, recon_loss, vq_loss = self.loss_function(recon_x, x, quantizer_output)

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
      self, recon_x: torch.Tensor, x: torch.Tensor, quantizer_output: dict
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none").sum(dim=-1)

    vq_loss = quantizer_output["loss"]

    return (
        (self.lamb * recon_loss + (1 - self.lamb) * vq_loss).mean(dim=0),
        recon_loss.mean(dim=0),
        vq_loss.mean(dim=0),
    )


class Quantizer(nn.Module):

  def __init__(self, model_config: dict) -> None:
    nn.Module.__init__(self)

    self.model_config = model_config

    self.embedding_dim = model_config["embedding_dim"]
    self.num_embeddings = model_config["num_embeddings"]
    self.commitment_loss_factor = model_config["commitment_loss_factor"]
    self.quantization_loss_factor = model_config["quantization_loss_factor"]

    self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    self.embeddings.weight.data.uniform_(-1 / self.num_embeddings,
                                         1 / self.num_embeddings)

  def forward(self, z: torch.Tensor) -> dict:
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

    output = {
        "quantized_vector": quantized,
        "quantized_indices": quantized_indices.unsqueeze(1),
        "loss": loss,
    }

    return output
