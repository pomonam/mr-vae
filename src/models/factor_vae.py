import math

import torch

from src.base_model import VAE
import torch.nn as nn


class FactorVAEDiscriminator(nn.Module):
  def __init__(self, latent_dim=16, hidden_units=1000) -> None:
    nn.Module.__init__(self)

    self.layers = nn.Sequential(
      nn.Linear(latent_dim, hidden_units),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_units, hidden_units),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_units, hidden_units),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_units, hidden_units),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_units, hidden_units),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_units, 2),
    )

  def forward(self, z: torch.Tensor):
    return self.layers(z)


class FactorVAE(VAE):

  def __init__(
      self,
      model_config,
      encoder = None,
      decoder = None,
  ):
    VAE.__init__(self, encoder=encoder, decoder=decoder)

    self.discriminator = FactorVAEDiscriminator(latent_dim=model_config.latent_dim)

    self.model_name = "FactorVAE"
    self.gamma = model_config.gamma

  def set_discriminator(self, discriminator) -> None:
    r"""This method is called to set the discriminator network
    Args:
        discriminator (BaseDiscriminator): The discriminator module that needs to be set to the model.
    """
    self.discriminator = discriminator


  def forward(self, inputs, **kwargs):
    """
    The VAE model
    Args:
        inputs (BaseDataset): The training dataset with labels
    Returns:
        ModelOutput: An instance of ModelOutput containing all the relevant parameters
    """

    # first batch
    x = inputs["data"]

    encoder_output = self.encoder(x)

    mu, log_var = encoder_output.embedding, encoder_output.log_covariance

    std = torch.exp(0.5 * log_var)
    z, _ = self._sample_gauss(mu, std)
    recon_x = self.decoder(z)["reconstruction"]

    # second batch
    x_bis = inputs["data_bis"]

    encoder_output = self.encoder(x_bis)

    mu_bis, log_var_bis = encoder_output.embedding, encoder_output.log_covariance

    std_bis = torch.exp(0.5 * log_var_bis)
    z_bis, _ = self._sample_gauss(mu_bis, std_bis)

    z_bis_permuted = self._permute_dims(z_bis).detach()

    recon_loss, autoencoder_loss, discriminator_loss = self.loss_function(
      recon_x, x, mu, log_var, z, z_bis_permuted
    )

    loss = autoencoder_loss + discriminator_loss

    output = ModelOutput(
      loss=loss,
      recon_loss=recon_loss,
      autoencoder_loss=autoencoder_loss,
      discriminator_loss=discriminator_loss,
      recon_x=recon_x,
      z=z,
      z_bis_permuted=z_bis_permuted,
    )

    return output


  def loss_function(self, recon_x, x, mu, log_var, z, z_bis_permuted):
    N = z.shape[0]  # batch size

    if self.model_config.reconstruction_loss == "mse":

      recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
      ).sum(dim=-1)

    elif self.model_config.reconstruction_loss == "bce":

      recon_loss = F.binary_cross_entropy(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
      ).sum(dim=-1)

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    latent_adversarial_score = self.discriminator(z)

    TC = (latent_adversarial_score[:, 0] - latent_adversarial_score[:, 1]).mean()
    autoencoder_loss = recon_loss + KLD + self.gamma * TC

    # discriminator loss
    permuted_latent_adversarial_score = self.discriminator(z_bis_permuted)

    true_labels = (
      torch.ones(N, requires_grad=False).type(torch.LongTensor).to(z.device)
    )
    fake_labels = (
      torch.zeros(N, requires_grad=False).type(torch.LongTensor).to(z.device)
    )

    TC_permuted = F.cross_entropy(
      latent_adversarial_score, fake_labels
    ) + F.cross_entropy(permuted_latent_adversarial_score, true_labels)

    discriminator_loss = 0.5 * TC_permuted

    return (
      (recon_loss).mean(dim=0),
      (autoencoder_loss).mean(dim=0),
      (discriminator_loss).mean(dim=0),
    )


  def _sample_gauss(self, mu, std):
    # Reparametrization trick
    # Sample N(0, I)
    eps = torch.randn_like(std)
    return mu + eps * std, eps


  def _permute_dims(self, z):
    permuted = torch.zeros_like(z)

    for i in range(z.shape[-1]):
      perms = torch.randperm(z.shape[0]).to(z.device)
      permuted[:, i] = z[perms, i]

    return permuted
