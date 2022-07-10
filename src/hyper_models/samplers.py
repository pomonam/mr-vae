import torch
from torch import nn
from src.hyper_models.modules import HyperModule


class BaseSampler(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def sample(outputs_dict):
        raise NotImplementedError


class HyperIsotropicGaussianSampler(nn.Module):
    def __init__(self, hidden_size, latent_size, hyper_type, block_name, bias=True):
        super().__init__()

        mean = nn.Linear(hidden_size, latent_size, bias=bias)
        log_stddev = nn.Linear(hidden_size, latent_size, bias=bias)

        self.mean = HyperModule(mean, hyper_type, block_name)
        self.log_stddev = HyperModule(log_stddev, hyper_type, block_name)

    def forward(self, x, beta, ignore_hyper=False):
        mean = self.mean(x, beta, ignore_hyper)
        log_stddev = self.log_stddev(x, beta, ignore_hyper)
        stddev = torch.exp(log_stddev)
        outputs_dict = {
            "mean": mean,
            "stddev": stddev,
            "ignore_hyper": ignore_hyper
        }
        return outputs_dict

    @staticmethod
    def sample(outputs_dict):
        if outputs_dict["ignore_hyper"]:
            return outputs_dict["mean"]
        else:
            eps = torch.randn_like(outputs_dict["stddev"])
            return eps.mul(outputs_dict["stddev"]).add_(outputs_dict["mean"])
