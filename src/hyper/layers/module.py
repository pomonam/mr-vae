import torch
from torch import nn

from src.config import HyperConfig


class HyperModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._net_inputs = None

    def set_net_inputs(self, value: torch.Tensor) -> None:
        self._net_inputs = value

    # def reset_beta(self) -> None:
    #     self._net_beta = None
