import math

import torch
from torch import nn

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
            self.encoder_trans = get_block(self.hyper_config.block_type)(1, hyper_config.preprocess_dim)
            self.decoder_trans = get_block(self.hyper_config.block_type)(1, hyper_config.preprocess_dim)
            self.sampler_trans = get_block(self.hyper_config.block_type)(1, hyper_config.preprocess_dim)

    def _register_hyper_modules(self):
        self._encoder_modules = self.encoder.register_hyper_modules()
        self._decoder_modules = self.decoder.register_hyper_modules()
        self._sampler_modules = self.sampler.register_hyper_modules()

    def set_beta(self, beta: torch.Tensor) -> None:
        if self.hyper_config.preprocess_beta:
            encoder_beta = self.encoder_trans(beta)
            self.encoder.set_beta(encoder_beta)

            decoder_beta = self.decoder_trans(beta)
            self.decoder.set_beta(decoder_beta)

            sampler_beta = self.sampler_trans(beta)
            self.sampler.set_beta(sampler_beta)

        else:
            self.encoder.set_beta(beta)
            self.decoder.set_beta(beta)
            self.sampler.set_beta(beta)

    def reset_beta(self) -> None:
        for hm in self._hyper_modules:
            hm.reset_beta()

    def sample_beta(self, x: torch.Tensor):
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}

        # These are useful constants ...
        const = math.sqrt(3)
        log_a_const = math.log(0.001)
        log_b_const = math.log(10)
        log_mid_const = (log_a_const + log_b_const) / 2
        diff_const = (log_mid_const - log_a_const)

        if self.hyper_config.sample_type == "fixed_log_uniform0.1":
            # sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(-const, const).to(device)
            # trans_beta = sample_dict["net_beta"] * (const / 3)
            # trans_beta = trans_beta * diff_const + log_mid_const
            # sample_dict["beta"] = torch.exp(trans_beta)
            const = math.sqrt(3)
            sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(-const, const).to(device)
            norm_beta = (sample_dict["net_beta"] * (2 * const) / 3) - 1
            sample_dict["beta"] = torch.pow(10, norm_beta)

        elif self.hyper_config.sample_type == "fixed_log_uniform1.0":
            sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).uniform_(-const, const).to(device)
            sample_dict["beta"] = torch.FloatTensor(batch_size, 1).to(device)
            sample_dict["beta"][sample_dict["net_beta"] >= 0.] = \
              torch.pow(10, sample_dict["net_beta"][sample_dict["net_beta"] >= 0.] * const / 3)
            sample_dict["beta"][sample_dict["net_beta"] < 0.] = \
              torch.pow(10, sample_dict["net_beta"][sample_dict["net_beta"] < 0.] * const)

        else:
            raise NotImplementedError

        return sample_dict

    def fixed_beta(self, x: torch.Tensor, trans_beta: float):
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}

        ones = torch.ones(batch_size, 1).to(device)
        beta = trans_beta * ones

        # These are useful constants ...
        const = math.sqrt(3)
        log_a_const = math.log(0.001)
        log_b_const = math.log(10)
        log_mid_const = (log_a_const + log_b_const) / 2
        diff_const = (log_mid_const - log_a_const)

        sample_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta

        if self.hyper_config.sample_type == "fixed_log_uniform0.1":
            # net_beta = (torch.log(beta) - log_mid_const) / diff_const
            # net_beta = net_beta * (3 / const)
            # sample_dict["net_beta"] = net_beta
            sample_dict["net_beta"] = (torch.log10(beta) + 1) * (3 / (2 * const))

        elif self.hyper_config.sample_type == "fixed_log_uniform1.0":
            if trans_beta >= 1.:
                sample_dict["net_beta"] = torch.log10(beta) * (3 / const)
            else:
                sample_dict["net_beta"] = torch.log10(beta) * (1 / const)

        else:
            raise NotImplementedError

        return sample_dict

    def sample_forward(self, x):
        sample_dict = self.sample_beta(x)
        self.set_beta(sample_dict["net_beta"])
        output_dict = self.forward(x)
        output_dict["beta"] = sample_dict["beta"]
        return output_dict

    def fixed_forward(self, x, beta):
        sample_dict = self.fixed_beta(x, beta)
        self.set_beta(sample_dict["net_beta"])
        output_dict = self.forward(x)
        output_dict["beta"] = sample_dict["beta"]
        return output_dict
