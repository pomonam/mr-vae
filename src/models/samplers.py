import torch
from torch import nn


class BaseSampler(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    def sample(self, x):
        raise NotImplementedError


class IsotropicGaussianSampler(nn.Module):
    def __init__(self, hidden_size, latent_size, bias=True):
        super().__init__()

        self.mean = nn.Linear(hidden_size, latent_size, bias=bias)
        self.log_stddev = nn.Linear(hidden_size, latent_size, bias=bias)

    def forward(self, x):
        mean = self.mean(x)
        log_stddev = self.log_stddev(x)
        stddev = torch.exp(log_stddev)
        return mean, stddev
