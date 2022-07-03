from decoder import MLPDecoder
from encoder import MLPEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):

    def __init__(self, encoder, decoder, beta=1.):
        super(BetaVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        mean, std = self.encoder(x)
        return mean, std

    def sample(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, std = self.encode(x)
        z = self.sample(mu, std)
        rx = self.decode(z)
        return rx, mu, std

