import math

import torch
from torch import nn

from src.config import HyperConfig
from src.hyper.layers.blocks import BatchNormResidualBlock
from src.hyper.layers.blocks import LinearBlock
from src.hyper.layers.blocks import MlpBlock
from src.hyper.layers.blocks import ResidualBlock
from src.hyper.layers.linear import HyperLinear
from src.hyper.layers.module import HyperModule
from src.hyper.transformations import stretch_sigmoid
from src.hyper.transformations import stretch_sigmoid_inv
# from src.hyper.layers.module import replace_module
from src.models.vae import BaseVae

_BLOCK_DICT = {
    "linear": LinearBlock,
    "mlp": MlpBlock,
    "residual": ResidualBlock,
    "bn_residual": BatchNormResidualBlock,
}


def replace_module(model: nn.Module, hyper_config: HyperConfig) -> None:
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_module(module, hyper_config)

        if isinstance(module, nn.Linear):
            hyper_module = HyperLinear(module, hyper_config)
            setattr(model, name, hyper_module)

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            hyper_module = HyperModule(module, hyper_config)
            setattr(model, name, hyper_module)


class HyperVae(BaseVae):

    def __init__(self, encoder, decoder, sampler, hyper_config):
        super().__init__(encoder, decoder, sampler)

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

        replace_module(self.encoder, hyper_config)
        replace_module(self.decoder, hyper_config)
        replace_module(self.sampler, hyper_config)
        self.hyper_config = hyper_config

        self._hyper_modules = []
        self._register_hyper_modules()

    def _register_hyper_modules(self):
        for module in self.encoder.modules():
            if isinstance(module, HyperModule):
                self._hyper_modules.append(module)

        for module in self.decoder.modules():
            if isinstance(module, HyperModule):
                self._hyper_modules.append(module)

        for module in self.sampler.modules():
            if isinstance(module, HyperModule):
                self._hyper_modules.append(module)

    def set_beta(self, beta: torch.Tensor) -> None:
        for hm in self._hyper_modules:
            hm.set_beta(beta)

    def reset_beta(self) -> None:
        for hm in self._hyper_modules:
            hm.reset_beta()

    def sample_beta(self, x: torch.Tensor):
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}

        if self.hyper_config.sample_type == "fixed_log_uniform0.1":
            const = math.sqrt(3)
            sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(-const, const).to(device)
            norm_beta = (sample_dict["net_beta"] * (2 * const) / 3) - 1
            sample_dict["trans_beta"] = torch.pow(10, norm_beta)

        elif self.hyper_config.sample_type == "fixed_log_uniform1.0":
            const = math.sqrt(3)
            sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(-const, const).to(device)
            sample_dict["trans_beta"] = torch.FloatTensor(batch_size, 1).to(device)
            sample_dict["trans_beta"][sample_dict["net_beta"] >= 0.] = \
                torch.pow(10, sample_dict["net_beta"][sample_dict["net_beta"] >= 0.] * const / 3)
            sample_dict["trans_beta"][sample_dict["net_beta"] < 0.] = \
                torch.pow(10, sample_dict["net_beta"][sample_dict["net_beta"] < 0.] * const)

        else:
            raise NotImplementedError

        return sample_dict

    def fixed_beta(self, x: torch.Tensor, trans_beta: float):
        batch_size = x.shape[0]
        device = x.device
        ones = torch.ones(batch_size, 1).to(device)
        beta = trans_beta * ones
        const = math.sqrt(3)

        if self.hyper_config.sample_type == "fixed_log_uniform0.1":
            net_beta = (torch.log10(beta) + 1) * (3 / (2 * const))

        elif self.hyper_config.sample_type == "fixed_log_uniform1.0":
            if trans_beta >= 1:
                net_beta = torch.log10(beta) * (3 / const)
            else:
                net_beta = torch.log10(beta) * (1 / const)

        else:
            raise NotImplementedError

        return net_beta

    def forward(self, x):
        raise NotImplementedError

    def sample_forward(self, x):
        sample_dict = self.sample_beta(x)
        self.set_beta(sample_dict["net_beta"])
        output_dict = self.forward(x)
        output_dict["beta"] = sample_dict["trans_beta"]
        return output_dict

    def fixed_forward(self, x, beta):
        net_beta = self.fixed_beta(x, beta)
        self.set_beta(net_beta)
        output_dict = self.forward(x)

        batch_size = x.shape[0]
        device = x.device
        output_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta
        return output_dict

    def hyper_ignore_forward(self, x):
        zero_beta = self.fixed_beta(x, 0)
        self.reset_beta()
        output_dict = self.forward(x)
        output_dict["beta"] = torch.zeros(zero_beta.shape)
        return output_dict
