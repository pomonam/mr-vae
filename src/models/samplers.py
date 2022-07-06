import torch
from torch import nn


class BaseSampler(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def sample(outputs_dict):
        raise NotImplementedError


class IsotropicGaussianSampler(BaseSampler):
    def __init__(self, hidden_size, latent_size, bias=True):
        super().__init__()

        self.mean = nn.Linear(hidden_size, latent_size, bias=bias)
        self.log_stddev = nn.Linear(hidden_size, latent_size, bias=bias)

    def forward(self, x, *argv):
        mean = self.mean(x, *argv)
        log_stddev = self.log_stddev(x, *argv)
        stddev = torch.exp(log_stddev)
        outputs_dict = {
            "mean": mean,
            "stddev": stddev
        }
        return outputs_dict

    @staticmethod
    def sample(outputs_dict):
        eps = torch.randn_like(outputs_dict["stddev"])
        return eps.mul(outputs_dict["stddev"]).add_(outputs_dict["mean"])
