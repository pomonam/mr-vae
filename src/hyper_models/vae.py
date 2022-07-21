import math

import torch

from src.hyper_models.transformations import stretch_sigmoid, stretch_sigmoid_inv
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

    def get_general_parameters(self):
        params = []
        for m in self._hyper_modules:
            params = params + list(m.parameters())
        return params

    def get_hyper_parameters(self):
        pass

    def set_beta(self, beta: torch.Tensor) -> None:
        for hm in self._hyper_modules:
            hm.set_beta(beta)

    def reset_beta(self) -> None:
        for hm in self._hyper_modules:
            hm.reset_beta()

    def sample_beta(self, x: torch.Tensor, warmup=False):
        batch_size = x.shape[0]
        device = x.device
        a = self.hyper_config.sample_range[0]
        b = self.hyper_config.sample_range[1]
        sample_dict = {}

        if self.hyper_config.sample_type == "log_uniform":
            a = math.log(a)
            b = math.log(b)
            sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(a, b).to(device)
            sample_dict["trans_beta"] = torch.exp(sample_dict["net_beta"])

        elif self.hyper_config.sample_type == "fixed_log_uniform":
            if warmup:
                sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(-2, 0.1).to(device)
                sample_dict["trans_beta"] = torch.pow(10, sample_dict["net_beta"] - 1)
            else:
                sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(-2, 2).to(device)
                # Equivalent to setting a = -3 and b = 1
                sample_dict["trans_beta"] = torch.pow(10, sample_dict["net_beta"] - 1)

        elif self.hyper_config.sample_type == "fixed_normal":
            sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).normal_(mean=0, std=1).to(device)
            sample_dict["trans_beta"] = stretch_sigmoid(sample_dict["net_beta"] - 2, low=1e-3, high=10, slope=2)

        elif self.hyper_config.sample_type == "uniform":
            sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(a, b).to(device)
            sample_dict["trans_beta"] = torch.exp(sample_dict["net_beta"])

        else:
            raise NotImplementedError

        return sample_dict

    def fixed_beta(self, x: torch.Tensor, beta: float):
        batch_size = x.shape[0]
        device = x.device
        ones = torch.ones(batch_size, 1).to(device)

        if self.hyper_config.sample_type == "log_uniform":
            beta = ones * beta
            trans_beta = torch.log(beta)

        elif self.hyper_config.sample_type == "fixed_log_uniform":
            beta = ones * beta
            trans_beta = torch.log10(beta) + 1

        elif self.hyper_config.sample_type == "fixed_normal":
            beta = ones * beta
            trans_beta = stretch_sigmoid_inv(beta, low=1e-3, high=10, slope=2) + 2

        elif self.hyper_config.sample_type == "normal":
            # sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).normal_(0, std=1).to(device)
            # sample_dict["trans_beta"] = stretch_sigmoid(sample_dict["net_beta"], low=a, high=b)
            trans_beta = None

        elif self.hyper_config.sample_type == "uniform":
            trans_beta = ones * math.log(beta)

        else:
            raise NotImplementedError

        return trans_beta

    def forward(self, x):
        raise NotImplementedError

    def sample_forward(self, x, warmup=False):
        sample_dict = self.sample_beta(x, warmup)
        self.set_beta(sample_dict["net_beta"])
        output_dict = self.forward(x)
        output_dict["beta"] = sample_dict["trans_beta"]
        return output_dict

    def fixed_forward(self, x, beta):
        fixed_beta = self.fixed_beta(x, beta)
        self.set_beta(fixed_beta)
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
