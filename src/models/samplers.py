import torch
from torch import nn


class BaseSampler(nn.Module):

    def forward(self, x: torch.Tensor) -> dict:
        raise NotImplementedError

    @staticmethod
    def sample(outputs_dict: dict, num_samples: int = 1) -> torch.Tensor:
        raise NotImplementedError


class IsotropicGaussianSampler(BaseSampler):
    def __init__(self, nh, nz):
        super().__init__()

        self.nh = nh
        self.nz = nz

        self.mean = nn.Linear(self.nh, self.nz)
        self.log_var = nn.Linear(self.nh, self.nz)

    def forward(self, x: torch.Tensor) -> dict:
        mean = self.mean(x)
        log_var = self.log_var(x)
        outputs_dict = {
            "mean": mean,
            "log_var": log_var
        }
        return outputs_dict

    @staticmethod
    def sample(outputs_dict, num_samples=1) -> torch.Tensor:
        # TODO(JB): Add support for num_samples > 1
        std = outputs_dict["log_var"].mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(outputs_dict["mean"])
