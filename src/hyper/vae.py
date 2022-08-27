import math

import numpy as np
import torch

from src.config import HyperConfig
from src.hyper.layers.blocks import BatchNormResidualBlock
from src.hyper.layers.blocks import get_block
from src.hyper.layers.blocks import LinearBlock
from src.hyper.layers.blocks import MlpBlock
from src.hyper.layers.blocks import ResidualBlock
from src.hyper.models import BaseHyperDecoder
from src.hyper.models import BaseHyperEncoder
from src.hyper.models import HyperIsotropicGaussianSampler
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

    def __init__(self,
                 encoder: BaseHyperEncoder,
                 decoder: BaseHyperDecoder,
                 sampler: HyperIsotropicGaussianSampler,
                 hyper_config: HyperConfig) -> None:
        super().__init__(encoder, decoder, sampler)

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.hyper_config = hyper_config

        if hyper_config.preprocess_beta:
            self.encoder_block = get_block(self.hyper_config.block_type)(
                in_features=1, width=hyper_config.preprocess_dim)
            self.decoder_block = get_block(self.hyper_config.block_type)(
                in_features=1, width=hyper_config.preprocess_dim)
            self.sampler_block = get_block(self.hyper_config.block_type)(
                in_features=1, width=hyper_config.preprocess_dim)

    def set_net_inputs(self, value: torch.Tensor) -> None:
        if self.hyper_config.preprocess_beta:
            encoder_inputs = self.encoder_block(value)
            self.encoder.set_net_inputs(encoder_inputs)
            decoder_inputs = self.decoder_block(value)
            self.decoder.set_net_inputs(decoder_inputs)
            sampler_inputs = self.sampler_block(value)
            self.sampler.set_net_inputs(sampler_inputs)
        else:
            self.encoder.set_net_inputs(value)
            self.decoder.set_net_inputs(value)
            self.sampler.set_net_inputs(value)

    def sample(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}

        if self.hyper_config.sample_type == "beta_log_uniform":
            sample_dict["net"] = torch.FloatTensor(batch_size, 1).uniform_(
                -_SQRT3, _SQRT3).to(device)
            beta = sample_dict["net"] * (_SQRT3 / 3)
            beta = beta * _LOG_DIFF + _LOG_M
            sample_dict["beta"] = torch.exp(beta)

        elif self.hyper_config.sample_type == "alpha_uniform":
            sample_dict["net"] = torch.FloatTensor(batch_size, 1).uniform_(
                -_SQRT3, _SQRT3).to(device)
            alpha = sample_dict["net"] * (_SQRT3 / 6)
            sample_dict["alpha"] = alpha + 0.5

        elif self.hyper_config.sample_type == "alpha_normal":
            sample_dict["net"] = torch.FloatTensor(batch_size,
                                                   1).normal_(0, 1).to(device)
            sample_dict["alpha"] = torch.sigmoid(sample_dict["net"])

        else:
            raise NotImplementedError

        return sample_dict

    def sample_inverse(self, x: torch.Tensor, value: float) -> dict:
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
            sample_dict["net"] = torch.log(sample_dict["alpha"] /
                                           (1 - sample_dict["alpha"]))

        else:
            raise NotImplementedError

        return sample_dict

    def sample_forward(self, x: torch.Tensor) -> dict:
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
            return np.linspace(0., 1., num=num, endpoint=False)

        raise NotImplementedError
