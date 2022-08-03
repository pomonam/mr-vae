import math

import torch
from torch import nn

from src.config import HyperConfig
from src.hyper.layers.blocks import BatchNormResidualBlock
from src.hyper.layers.blocks import get_block
from src.hyper.layers.blocks import LinearBlock
from src.hyper.layers.blocks import MlpBlock
from src.hyper.layers.blocks import ResidualBlock
from src.hyper.layers.linear import HyperLinear
from src.hyper.layers.module import HyperModule
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

        if isinstance(module, nn.Conv2d) or isinstance(module,
                                                       nn.ConvTranspose2d):
            hyper_module = HyperModule(module, hyper_config)
            setattr(model, name, hyper_module)


class HyperVae(BaseVae):
    def __init__(self, encoder, decoder, sampler, hyper_config: HyperConfig):
        super().__init__(encoder, decoder, sampler)

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

        replace_module(self.encoder, hyper_config)
        replace_module(self.decoder, hyper_config)
        replace_module(self.sampler, hyper_config)
        self.hyper_config = hyper_config

        self._hyper_modules = []
        self._encoder_modules = []
        self._sampler_modules = []
        self._decoder_modules = []
        self._register_hyper_modules()

        if hyper_config.preprocess_beta:
            self.encoder_trans = get_block(self.hyper_config.block_type)(
                1, hyper_config.preprocess_dim, False)
            self.decoder_trans = get_block(self.hyper_config.block_type)(
                1, hyper_config.preprocess_dim, False)
            self.sampler_trans = get_block(self.hyper_config.block_type)(
                1, hyper_config.preprocess_dim, False)

    def _register_hyper_modules(self):
        for module in self.encoder.modules():
            if isinstance(module, HyperModule):
                self._encoder_modules.append(module)
                self._hyper_modules.append(module)

        for module in self.decoder.modules():
            if isinstance(module, HyperModule):
                self._decoder_modules.append(module)
                self._hyper_modules.append(module)

        for module in self.sampler.modules():
            if isinstance(module, HyperModule):
                self._sampler_modules.append(module)
                self._hyper_modules.append(module)

    def set_beta(self, beta: torch.Tensor) -> None:
        if self.hyper_config.preprocess_beta:
            encoder_beta = self.encoder_trans(beta)
            for hm in self._encoder_modules:
                hm.set_beta(encoder_beta)

            decoder_beta = self.decoder_trans(beta)
            for hm in self._decoder_modules:
                hm.set_beta(decoder_beta)

            sampler_beta = self.sampler_trans(beta)
            for hm in self._sampler_modules:
                hm.set_beta(sampler_beta)
        else:
            for hm in self._hyper_modules:
                hm.set_beta(beta)

    def reset_beta(self) -> None:
        for hm in self._hyper_modules:
            hm.reset_beta()

    def sample_beta(self, x: torch.Tensor):
        batch_size = x.shape[0]
        device = x.device
        sample_dict = {}
        const = math.sqrt(3)

        if self.hyper_config.sample_type == "fixed_log_uniform0.1":
            sample_dict["net_beta"] = torch.FloatTensor(
                batch_size, 1).uniform_(-const, const).to(device)
            trans_beta = (sample_dict["net_beta"] * (2 * const) / 3) - 1
            sample_dict["trans_beta"] = torch.pow(10, trans_beta)

        elif self.hyper_config.sample_type == "fixed_log_uniform1.0":
            sample_dict["net_beta"] = torch.FloatTensor(
                batch_size, 1).uniform_(-const, const).to(device)
            sample_dict["trans_beta"] = torch.FloatTensor(batch_size,
                                                          1).to(device)
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
