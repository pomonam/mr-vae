import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_activation


class IsotropicGaussian(nn.Module):
    def __init__(self, latent_size, bias=True):
        super().__init__()

        self.mean = nn.Linear(latent_size, latent_size, bias=bias)
        self.log_stddev = nn.Linear(latent_size, latent_size, bias=bias)

    def forward(self, x):
        mean = self.mean(x)
        log_stddev = self.log_stddev(x)
        stddev = torch.exp(log_stddev)
        return mean, stddev
