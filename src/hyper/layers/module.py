import torch
from torch import nn

from src.config import HyperConfig


class HyperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._net_beta = None

    def set_beta(self, beta: torch.Tensor) -> None:
        self._net_beta = beta

    def reset_beta(self) -> None:
        self._net_beta = None
