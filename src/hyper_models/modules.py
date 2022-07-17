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
        if self.hyper_type not in ["add", "mult"]:
            raise ValueError("Invalid hyper_type {}".format(str(self.hyper_type)))
        self.block_type = hyper_config.block_type

        if isinstance(module, nn.Linear):
            # Create a copy of a module ...
            self.module = copy.deepcopy(module)
            self.hyper_module = _BLOCK_DICT[self.block_type](module.out_features)

            def _hyper_mult(a, b):
                return a * b
            self.hyper_mult = _hyper_mult

        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            self.module = copy.deepcopy(module)
            self.hyper_module = _BLOCK_DICT[self.block_type](module.out_channels)

            def _hyper_mult(a, b):
                return a * b.unsqueeze(-1).unsqueeze(-1)
            self.hyper_mult = _hyper_mult

        if self.hyper_type == "add":
            self.add_hyper_module = copy.deepcopy(module)
        else:
            self.add_hyper_module = None

        self._beta = None

    def set_beta(self, beta: torch.Tensor) -> None:
        self._beta = beta

    def reset_beta(self) -> None:
        self._beta = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.hyper_type == "add":
            if self._beta is None:
                out = self.module(inputs)
            else:
                add_out = self.add_hyper_module(inputs)
                hyper_out = self.hyper_module(self._beta)
                hyper_out = self.hyper_mult(add_out, hyper_out)
                out = self.module(inputs) + hyper_out

        elif self.hyper_type == "mult":
            if self._beta is None:
                print("Warning: hyper_type mult does not support ignore_hyper")
            hyper_out = self.hyper_module(self._beta)
            orig_out = self.module(inputs)
            out = self.hyper_mult(orig_out, hyper_out)

        else:
            raise NotImplementedError

        return out
