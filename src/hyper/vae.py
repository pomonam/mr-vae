import math

import torch
from torch import nn
import numpy as np

from src.config import HyperConfig
from src.hyper.layers.blocks import BatchNormResidualBlock
from src.hyper.layers.blocks import get_block
from src.hyper.layers.blocks import LinearBlock
from src.hyper.layers.blocks import MlpBlock
from src.hyper.layers.blocks import ResidualBlock
from src.models.vae import BaseVae

_BLOCK_DICT = {
    "linear": LinearBlock,
    "mlp": MlpBlock,
    "residual": ResidualBlock,
    "bn_residual": BatchNormResidualBlock,
}

_SQRT3 = math.sqrt(3)
_LOG_A = math.log(0.001)
_LOG_B = math.log(10)
_LOG_M = (_LOG_A + _LOG_B) / 2
_LOG_DIFF = (_LOG_M - _LOG_A)


class HyperVae(BaseVae):
    def __init__(self, encoder, decoder, sampler, hyper_config: HyperConfig):
        super().__init__(encoder, decoder, sampler)

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

        self.hyper_config = hyper_config

        self._hyper_modules = []
        self._encoder_modules = None
        self._sampler_modules = None
        self._decoder_modules = None
        self._register_hyper_modules()

        if hyper_config.preprocess_beta:
            self.encoder_block = get_block(self.hyper_config.block_type)(1, hyper_config.preprocess_dim)
            self.decoder_block = get_block(self.hyper_config.block_type)(1, hyper_config.preprocess_dim)
            self.sampler_block = get_block(self.hyper_config.block_type)(1, hyper_config.preprocess_dim)

    def _register_hyper_modules(self):
        self._encoder_modules = self.encoder.register_hyper_modules()
        self._decoder_modules = self.decoder.register_hyper_modules()
        self._sampler_modules = self.sampler.register_hyper_modules()

    def set_net_inputs(self, beta: torch.Tensor) -> None:
        if self.hyper_config.preprocess_beta:
            encoder_beta = self.encoder_block(beta)
            self.encoder.set_net_inputs(encoder_beta)

            decoder_beta = self.decoder_block(beta)
            self.decoder.set_net_inputs(decoder_beta)

            sampler_beta = self.sampler_block(beta)
            self.sampler.set_net_inputs(sampler_beta)

        else:
            self.encoder.set_net_inputs(beta)
            self.decoder.set_net_inputs(beta)
            self.sampler.set_net_inputs(beta)

    def sample(self, x: torch.Tensor):
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}

        if self.hyper_config.sample_type == "beta_log_uniform":
            sample_dict["net"] = torch.FloatTensor(batch_size, 1).uniform_(-_SQRT3, _SQRT3).to(device)
            beta = sample_dict["net"] * (_SQRT3 / 3)
            beta = beta * _LOG_DIFF + _LOG_M
            sample_dict["beta"] = torch.exp(beta)

        elif self.hyper_config.sample_type == "alpha_uniform":
            sample_dict["net"] = torch.FloatTensor(batch_size, 1).uniform_(-_SQRT3, _SQRT3).to(device)
            alpha = sample_dict["net"] * (_SQRT3 / 6)
            sample_dict["alpha"] = alpha + 0.5

        elif self.hyper_config.sample_type == "alpha_normal":
            sample_dict["net"] = torch.FloatTensor(batch_size, 1).normal_(0, 1).to(device)
            sample_dict["alpha"] = torch.sigmoid(sample_dict["net"])

        else:
            raise NotImplementedError

        return sample_dict

    def sample_inverse(self, x: torch.Tensor, value: float):
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}

        ones = torch.ones(batch_size, 1).to(device)

        if self.hyper_config.sample_type == "beta_log_uniform":
            beta = value * ones
            sample_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta
            net_beta = (torch.log(sample_dict["beta"]) - _LOG_M) / _LOG_DIFF
            sample_dict["net"] = net_beta * (3 / _SQRT3)

        elif self.hyper_config.sample_type == "alpha_uniform":
            alpha = value * ones
            sample_dict["alpha"] = torch.ones(batch_size, 1).to(device) * alpha
            net_alpha = sample_dict["alpha"] - 0.5
            sample_dict["net"] = net_alpha * (6 / _SQRT3)

        elif self.hyper_config.sample_type == "alpha_normal":
            alpha = value * ones
            sample_dict["alpha"] = torch.ones(batch_size, 1).to(device) * alpha
            sample_dict["net"] = torch.log(sample_dict["alpha"] / (1 - sample_dict["alpha"]))

        else:
            raise NotImplementedError

        return sample_dict

    def sample_forward(self, x):
        sample_dict = self.sample(x)
        self.set_net_inputs(sample_dict["net"])
        output_dict = self.forward(x)

        if "beta" in self.hyper_config.sample_type:
            output_dict["beta"] = sample_dict["beta"]
        if "alpha" in self.hyper_config.sample_type:
            output_dict["alpha"] = sample_dict["alpha"]

        return output_dict

    def inverse_forward(self, x, value):
        sample_dict = self.sample_inverse(x, value)
        self.set_net_inputs(sample_dict["net"])
        output_dict = self.forward(x)

        if "beta" in self.hyper_config.sample_type:
            output_dict["beta"] = sample_dict["beta"]
        if "alpha" in self.hyper_config.sample_type:
            output_dict["alpha"] = sample_dict["alpha"]

        return output_dict

    def get_test_samples(self, num=20):
        if "beta" in self.hyper_config.sample_type:
            return np.logspace(-3, 1, num=num, base=10)

        if "alpha" in self.hyper_config.sample_type:
            return np.linspace(0.05, 0.95, num=num)

        raise NotImplementedError
