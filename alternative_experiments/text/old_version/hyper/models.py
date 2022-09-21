import torch
from torch import nn

from alternative_experiments.text.old_version.hyper.layers.linear import HyperLinear
from alternative_experiments.text.old_version.hyper.layers.module import HyperModule
from alternative_experiments.text.old_version.models.base_decoder import BaseDecoder
from alternative_experiments.text.old_version.models.base_encoder import BaseEncoder
from alternative_experiments.text.old_version.models.samplers import BaseSampler


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


class BaseHyperDecoder(HyperStructure, BaseDecoder):

    def __init__(self) -> None:
        super().__init__()


class HyperIsotropicGaussianSampler(BaseSampler):

    def __init__(self, nh: int, nz: int) -> None:
        super().__init__()
        self.nh = nh
        self.nz = nz

        self.mean = HyperLinear(self.nh, self.nz)
        self.log_var = HyperLinear(self.nh, self.nz)

    def forward(self, x: torch.Tensor) -> dict:
        mean = self.mean(x)
        log_var = self.log_var(x)
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
        # self.mean_proj.set_net_inputs(value)
        # self.log_var_proj.set_net_inputs(value)
