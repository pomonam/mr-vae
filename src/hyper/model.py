import copy

import torch
from torch import nn

from src.config import HyperConfig
from src.hyper.layers.blocks import BatchNormResidualBlock
from src.hyper.layers.blocks import LinearBlock
from src.hyper.layers.blocks import MlpBlock
from src.hyper.layers.blocks import ResidualBlock
from src.hyper.layers.linear import HyperLinear

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

        # if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        #     hyper_module = HyperModule(module, hyper_config)
        #     setattr(model, name, hyper_module)
