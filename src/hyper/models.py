import torch
from torch import nn

from src.hyper.layers.linear import HyperLinear
from src.hyper.layers.module import HyperModule
from src.models.samplers import BaseSampler
from src.models.base_decoder import BaseDecoder
from src.models.base_encoder import BaseEncoder
from src.config import HyperConfig


class HyperStructure(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def set_net_inputs(self, value: torch.Tensor) -> None:
        for module in self.modules():
            if isinstance(module, HyperModule):
                module.set_net_inputs(value)

    def reset_net_inputs(self) -> None:
        for module in self.modules():
            if isinstance(module, HyperModule):
                module.reset_net_inputs()


class BaseHyperEncoder(HyperStructure, BaseEncoder):

    def __init__(self) -> None:
        super().__init__()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     raise NotImplementedError
    #
    # def encode(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.forward(x)


class BaseHyperDecoder(HyperStructure, BaseDecoder):

    def __init__(self) -> None:
        super().__init__()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     raise NotImplementedError
    #
    # def decode(self, z: torch.Tensor) -> torch.Tensor:
    #     raise self.forward(z)


class HyperIsotropicGaussianSampler(BaseSampler):

    def __init__(self, nh: int, nz: int, hyper_config: HyperConfig) -> None:
        super().__init__()
        self.nh = nh
        self.nz = nz

        self.mean = HyperLinear(self.nh, self.nh, "none", hyper_config)
        self.log_var = HyperLinear(self.nh, self.nh, "none", hyper_config)
        # self.mean = nn.Linear(self.nh, self.nz)
        # self.log_var = nn.Linear(self.nh, self.nz)

    def forward(self, x: torch.Tensor) -> dict:
        mean = self.mean(x)
        # mean = self.mean2(mean)

        log_var = self.log_var(x)
        # log_var = self.log_var2(log_var)

        outputs_dict = {"mean": mean, "log_var": log_var}
        return outputs_dict

    @staticmethod
    def sample(outputs_dict, num_samples=1) -> torch.Tensor:
        std = outputs_dict["log_var"].mul(0.5).exp()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(outputs_dict["mean"])

    def set_net_inputs(self, value: torch.Tensor) -> None:
        self.mean.set_net_inputs(value)
        self.log_var.set_net_inputs(value)
