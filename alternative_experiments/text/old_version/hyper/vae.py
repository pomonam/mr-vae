import math

import numpy as np
import torch
from alternative_experiments.text.old_version.hyper.models import BaseHyperDecoder, BaseHyperEncoder, HyperIsotropicGaussianSampler
from src.config import HyperConfig
from alternative_experiments.text.old_version.hyper.layers.blocks import BatchNormResidualBlock
from alternative_experiments.text.old_version.hyper.layers.blocks import LinearBlock
from alternative_experiments.text.old_version.hyper.layers.blocks import MlpBlock
from alternative_experiments.text.old_version.hyper.layers.blocks import ResidualBlock
from alternative_experiments.text.old_version.models.vae import BaseVae

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

    def set_net_inputs(self, value: torch.Tensor) -> None:
        self.encoder.set_net_inputs(value)
        self.decoder.set_net_inputs(value)
        self.sampler.set_net_inputs(value)

    def sample(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}
        sample_dict["net"] = torch.FloatTensor(batch_size, 1).uniform_(
            -_SQRT3, _SQRT3).to(device)
        beta = sample_dict["net"] * (_SQRT3 / 3)
        beta = beta * _LOG_DIFF + _LOG_M
        sample_dict["beta"] = torch.exp(beta)
        return sample_dict

    def sample_inverse(self, x: torch.Tensor, value: float) -> dict:
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}
        ones = torch.ones(batch_size, 1).to(device)
        beta = value * ones
        sample_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta
        net_beta = (torch.log(sample_dict["beta"]) - _LOG_M) / _LOG_DIFF
        sample_dict["net"] = net_beta * (3 / _SQRT3)
        return sample_dict

    def sample_forward(self, x: torch.Tensor) -> dict:
        sample_dict = self.sample(x)
        self.set_net_inputs(sample_dict["net"])
        output_dict = self.forward(x)
        output_dict["beta"] = sample_dict["beta"]
        return output_dict

    def inverse_forward(self, x, value):
        sample_dict = self.sample_inverse(x, value)
        self.set_net_inputs(sample_dict["net"])
        output_dict = self.forward(x)
        output_dict["beta"] = sample_dict["beta"]

        return output_dict

    def get_test_samples(self, num=20):
        return np.logspace(-3, 1, num=num, base=10)
