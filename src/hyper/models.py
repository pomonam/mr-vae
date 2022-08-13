from src.hyper.layers.module import HyperModule
from torch import nn
from src.models.samplers import BaseSampler
import torch
from src.hyper.layers.linear import HyperLinear


class HyperStructure(nn.Module):
    def __init__(self):
        super().__init__()
        self._hyper_modules = []

    def register_hyper_modules(self):
        for module in self.modules():
            if isinstance(module, HyperModule):
                self._hyper_modules.append(module)
        return self._hyper_modules

    def set_beta(self, beta: torch.Tensor) -> None:
        for hm in self._hyper_modules:
            hm.set_beta(beta)


class BaseHyperEncoder(HyperStructure):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class BaseHyperDecoder(HyperStructure):
    require_inputs = False

    def __init__(self):
        super().__init__()
        self._hyper_modules = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise self.forward(z)

    def reconstruct_error(self, x: torch.Tensor, z: torch.Tensor,
                          *argv) -> torch.Tensor:
        raise NotImplementedError


class HyperIsotropicGaussianSampler(BaseSampler):
    def __init__(self, nh, nz, hyper_config):
        super().__init__()
        self.nh = nh
        self.nz = nz

        self.mean1 = nn.Linear(self.nh, self.nh)
        self.log_var1 = nn.Linear(self.nh, self.nh)

        self.hyper_mean = HyperLinear(self.nh, hyper_config)
        self.hyper_log_var = HyperLinear(self.nh, hyper_config)

        self.mean2 = nn.Linear(self.nh, self.nz)
        self.log_var2 = nn.Linear(self.nh, self.nz)

        self._hyper_modules = [self.hyper_mean, self.hyper_log_var]

    def forward(self, x: torch.Tensor) -> dict:
        mean = self.mean1(x)
        mean = self.hyper_mean(mean)
        mean = self.mean2(mean)

        log_var = self.log_var1(x)
        log_var = self.hyper_log_var(log_var)
        log_var = self.log_var2(log_var)

        outputs_dict = {"mean": mean, "log_var": log_var}
        return outputs_dict

    @staticmethod
    def sample(outputs_dict, num_samples=1) -> torch.Tensor:
        std = outputs_dict["log_var"].mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(outputs_dict["mean"])

    def register_hyper_modules(self):
        return self._hyper_modules

    def set_beta(self, beta: torch.Tensor) -> None:
        self.hyper_mean.set_beta(beta)
        self.hyper_log_var.set_beta(beta)
