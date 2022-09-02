import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder


class BaseAE(nn.Module):

  def __init__(self,
               encoder: BaseEncoder = None,
               decoder: BaseDecoder = None,
               reconstruction_loss: str = "mse"):

    nn.Module.__init__(self)

    self.model_name = "BaseAE"
    self.encoder = encoder
    self.decoder = decoder
    self.reconstruction_loss = reconstruction_loss

  def forward(self, inputs, **kwargs):
    raise NotImplementedError()

  def reconstruct(self, inputs: torch.Tensor):
    return self({"data": inputs, "data_bis": inputs}).recon_x

  def interpolate(
      self,
      starting_inputs: torch.Tensor,
      ending_inputs: torch.Tensor,
      granularity: int = 10,
  ):
    assert starting_inputs.shape[0] == ending_inputs.shape[0], (
        "The number of starting_inputs should equal the number of ending_inputs. Got "
        f"{starting_inputs.shape[0]} sampler for starting_inputs and {ending_inputs.shape[0]} "
        "for endinging_inputs."
    )

    starting_z = self({"data": starting_inputs, "data_bis": starting_inputs}).z
    ending_z = self({"data": ending_inputs, "data_bis": ending_inputs}).z
    t = torch.linspace(0, 1, granularity).to(starting_inputs.device)
    intep_line = (torch.kron(
        starting_z.reshape(starting_z.shape[0], -1),
        (1 - t).unsqueeze(-1)) + torch.kron(
            ending_z.reshape(ending_z.shape[0], -1),
            t.unsqueeze(-1))).reshape((starting_z.shape[0] * t.shape[0],) +
                                      (starting_z.shape[1:]))

    decoded_line = self.decoder(intep_line).reconstruction.reshape((
        starting_inputs.shape[0],
        t.shape[0],
    ) + (starting_inputs.shape[1:]))

    return decoded_line


class VAE(BaseAE):

  def __init__(self,
               encoder: BaseEncoder = None,
               decoder: BaseDecoder = None,
               reconstruction_loss: str = "mse"):

    BaseAE.__init__(
        self,
        encoder=encoder,
        decoder=decoder,
        reconstruction_loss=reconstruction_loss)

    self.model_name = "VAE"

  def _sample_gauss(self, mu, std):
    # Reparametrization trick
    # Sample N(0, I)
    eps = torch.randn_like(std)
    return mu + eps * std, eps

  def get_nll(self, data, n_samples=1, batch_size=100):

    if n_samples <= batch_size:
      n_full_batch = 1
    else:
      n_full_batch = n_samples // batch_size
      n_samples = batch_size

    log_p = []

    for i in range(len(data)):
      x = data[i].unsqueeze(0)

      log_p_x = []

      for j in range(n_full_batch):

        x_rep = torch.cat(batch_size * [x])

        encoder_output = self.encoder(x_rep)
        mu, log_var = encoder_output["embedding"], encoder_output["log_covariance"]

        std = torch.exp(0.5 * log_var)
        z, _ = self._sample_gauss(mu, std)

        log_q_z_given_x = -0.5 * (log_var +
                                  (z - mu)**2 / torch.exp(log_var)).sum(dim=-1)
        log_p_z = -0.5 * (z**2).sum(dim=-1)

        recon_x = self.decoder(z)["reconstruction"]

        if self.reconstruction_loss == "mse":

          log_p_x_given_z = -0.5 * F.mse_loss(
              recon_x.reshape(x_rep.shape[0], -1),
              x_rep.reshape(x_rep.shape[0], -1),
              reduction="none",
          ).sum(dim=-1) - torch.tensor([
              np.prod(self.input_dim) / 2 * np.log(np.pi * 2)
          ]).to(data.device
               )  # decoding distribution is assumed unit variance  N(mu, I)

        elif self.reconstruction_loss == "bce":

          log_p_x_given_z = -F.binary_cross_entropy(
              recon_x.reshape(x_rep.shape[0], -1),
              x_rep.reshape(x_rep.shape[0], -1),
              reduction="none",
          ).sum(dim=-1)

        log_p_x.append(log_p_x_given_z + log_p_z -
                       log_q_z_given_x)  # log(2*pi) simplifies

      log_p_x = torch.cat(log_p_x)

      log_p.append((torch.logsumexp(log_p_x, 0) - np.log(len(log_p_x))).item())
    return np.mean(log_p)
