import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# from ...data.datasets import BaseDataset
# from ..base.base_utils import ModelOutput
# from ..nn import BaseDecoder, BaseEncoder
# from ..vae import VAE
# from .beta_vae_config import BetaVAEConfig


# class BetaVAE(VAE):
class BetaVAE(nn.Module):

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

    def _sample_gauss(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std, eps
