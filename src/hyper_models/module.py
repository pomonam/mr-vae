import copy

import torch
from torch import nn

from src.config import HyperConfig
from src.hyper_models.blocks import BatchNormResidualBlock
from src.hyper_models.blocks import LinearBlock
from src.hyper_models.blocks import MlpBlock
from src.hyper_models.blocks import ResidualBlock

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
            hyper_module = HyperModule(module, hyper_config)
            setattr(model, name, hyper_module)

        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            hyper_module = HyperModule(module, hyper_config)
            setattr(model, name, hyper_module)


class HyperModule(nn.Module):
    def __init__(self,
                 module: nn.Module,
                 hyper_config: HyperConfig):
        super().__init__()

        self.hyper_type = hyper_config.hyper_type
        if self.hyper_type not in ["add", "mult", "svd"]:
            raise ValueError("Invalid hyper_type {}".format(str(self.hyper_type)))
        self.block_type = hyper_config.block_type
        self.include_output_layer = hyper_config.include_output_layer
        self._beta = None

    def set_beta(self, beta: torch.Tensor) -> None:
        self._beta = beta

    def reset_beta(self) -> None:
        self._beta = None

