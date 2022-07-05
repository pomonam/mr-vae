import torch
from torch import nn
import torchvision


class BaseVae(nn.Module):

    def __init__(self, encoder, decoder, sampler):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

    def encode(self, x):
        ix = self.encoder(x)
        mean, std = self.sampler(ix)
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
        outputs_dict = {
            "inputs": x,
            "mean": mu,
            "stddev": std,
            "logits": rx
        }
        return outputs_dict

    def sample_inputs(self, batch):
        outputs_dict = self.model(batch["inputs"])
        recons = outputs_dict["logits"]
        torchvision.utils.save_image(recons)


class HyperVae(nn.Module):

    def __init__(self, encoder, decoder, sampler):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

    def encode(self, x, beta):
        ix = self.encoder(x, beta)
        mean, std = self.sampler(ix)
        return mean, std

    def sample(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z, beta):
        return self.decoder(z, beta)

    def forward(self, x, beta):
        mu, std = self.encode(x, beta)
        z = self.sample(mu, std)
        rx = self.decode(z, beta)
        outputs_dict = {
            "inputs": x,
            "mean": mu,
            "stddev": std,
            "logits": rx
        }
        return outputs_dict

    def sample_inputs(self, batch):
        outputs_dict = self.model(batch["inputs"])
        recons = outputs_dict["logits"]
        torchvision.utils.save_image(recons)
