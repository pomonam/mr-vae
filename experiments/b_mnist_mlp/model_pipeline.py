import torch
from torch import nn

from src.criterions import binary_cross_entropy
from src.criterions import kl_gaussian
from src.models.decoders import MLPDecoder
from src.models.encoders import MLPEncoder
from src.models.samplers import IsotropicGaussianSampler
from src.models.vae import BaseVae, HyperVae
from src.models.hyper import replace_module


class BinarizedMnistMlpModel(BaseVae):
    def encode(self, x):
        ix = self.encoder(x)
        mean, std = self.sampler(ix)
        return mean, std

    def sample(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

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


class HyperBinarizedMnistMlpModel(HyperVae):
    def __init__(self, encoder, decoder, sampler, block_name):
        super().__init__(encoder, decoder, sampler)

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        replace_module(self.encoder, block_name)
        replace_module(self.decoder, block_name)
        replace_module(self.sampler, block_name)

    def encode(self, x, beta):
        ix = self.encoder(x, beta)
        mean, std = self.sampler(ix, beta)
        return mean, std

    def sample(self, mu, std):
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

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


def build_model(device):
    # The architecture is inspired from:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py
    encoder = MLPEncoder(structure=(784, 512))
    sampler = IsotropicGaussianSampler(hidden_size=512, latent_size=10)
    decoder = MLPDecoder(structure=(10, 512, 784))
    model = BinarizedMnistMlpModel(encoder=encoder, decoder=decoder, sampler=sampler)
    return model.to(device)


def build_hyper_model(block_name, device):
    # The architecture is inspired from:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/vae.py
    encoder = MLPEncoder(structure=(784, 512))
    sampler = IsotropicGaussianSampler(hidden_size=512, latent_size=10)
    decoder = MLPDecoder(structure=(10, 512, 784))
    model = HyperBinarizedMnistMlpModel(encoder=encoder, decoder=decoder, sampler=sampler, block_name=block_name)
    return model.to(device)


class BinarizedMnistMlpCriterion(nn.Module):
    # def __init__(self, beta):
    #     super().__init__()
    #     self.beta = beta

    def get_metric_lst(self):
        return ["loss", "rate", "distortion"]

    def forward(self, outputs_dict, beta):
        log_likelihood = -binary_cross_entropy(outputs_dict["inputs"], outputs_dict["logits"])
        kl = kl_gaussian(outputs_dict["mean"], torch.square(outputs_dict["stddev"]))
        elbo = log_likelihood - beta * kl
        loss_dict = {
            "loss": -torch.mean(elbo).item(),
            "distortion": -torch.mean(log_likelihood).item(),
            "rate": torch.mean(kl).item()
        }
        return -torch.mean(elbo), loss_dict


def build_criterion(device):
    loss_fnc = BinarizedMnistMlpCriterion()
    return loss_fnc.to(device)
