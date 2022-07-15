import math

import torch

from src.hyper_models.modules import HyperModule
from src.hyper_models.modules import replace_module
from src.models.vae import BaseVae


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
        a = self.hyper_config.sample_range[0]
        b = self.hyper_config.sample_range[1]

        if self.hyper_config.sample_type == "log_uniform":
            a = math.log(a)
            b = math.log(b)
            return torch.FloatTensor(batch_size, 1).uniform_(a, b).to(device)

        elif self.hyper_config.sample_type == "uniform":
            return torch.FloatTensor(batch_size, 1).uniform_(a, b).to(device)

        else:
            raise NotImplementedError

    def fixed_beta(self, x: torch.Tensor, beta: float):
        batch_size = x.shape[0]
        device = x.device
        return torch.ones(batch_size, 1).to(device) * beta

    def forward(self, x):
        raise NotImplementedError

    def sample_forward(self, x):
        sample_beta = self.sample_beta(x)
        self.set_beta(sample_beta)
        output_dict = self.forward(x)
        output_dict["beta"] = sample_beta
        return output_dict

    def fixed_forward(self, x, beta):
        fixed_beta = self.fixed_beta(x, beta)
        self.set_beta(fixed_beta)
        output_dict = self.forward(x)
        output_dict["beta"] = fixed_beta
        return output_dict

    def hyper_ignore_forward(self, x):
        zero_beta = self.fixed_beta(x, 0)
        self.reset_beta()
        output_dict = self.forward(x)
        output_dict["beta"] = zero_beta
        return output_dict
