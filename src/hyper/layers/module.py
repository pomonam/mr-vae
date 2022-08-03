import torch
from torch import nn

from src.config import HyperConfig


class HyperModule(nn.Module):
    def __init__(self, module: nn.Module, hyper_config: HyperConfig):
        super().__init__()

        self.cfg = hyper_config
        self.hyper_type = self.cfg.hyper_type
        if self.hyper_type not in ["add", "s_add", "mult"]:
            raise ValueError("Invalid hyper_type {}".format(
                str(self.hyper_type)))

        self._beta = None

    def set_beta(self, beta: torch.Tensor) -> None:
        self._beta = beta

    def reset_beta(self) -> None:
        self._beta = None
