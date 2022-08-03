import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
