import torch
from torch import nn

from src.config import HyperConfig


def get_block(name: str):
    _BLOCK_DICT = {
        "linear": LinearBlock,
        "mlp": MlpBlock,
        "residual": ResidualBlock,
        "bn_residual": BatchNormResidualBlock,
    }
    return _BLOCK_DICT[name]


class BaseBlock(nn.Module):

    def __init__(self,
                 width: int,
                 hyper_config: HyperConfig):
        super().__init__()
        self.width = width
        self.include_sigmoid_activation = hyper_config.include_sigmoid_activation

        self.layers = None
        self._construct_layers()

    def _construct_layers(self) -> None:
        raise NotImplementedError

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LinearBlock(BaseBlock):
    def _construct_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(1, self.width)
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.layers(beta)
        if self.include_sigmoid_activation:
            out = torch.sigmoid(out)
        return out


class MlpBlock(BaseBlock):
    def _construct_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(1, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
            nn.GELU(),
            nn.Linear(self.width, self.width),
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.layers(beta)
        if self.include_sigmoid_activation:
            out = torch.sigmoid(out)
        return out


class ResidualBlock(BaseBlock):

    def _construct_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
        )
        self.temp_layer = nn.Sequential(
            nn.Linear(1, self.width, bias=True),
            nn.ReLU()
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.temp_layer(beta)
        out = out + self.layers(out)
        if self.include_sigmoid_activation:
            out = torch.sigmoid(out)
        return out


class BatchNormResidualBlock(BaseBlock):

    def _construct_layers(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(self.width, self.width, bias=False),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
            nn.BatchNorm1d(self.width),
            nn.ReLU(),
            nn.Linear(self.width, self.width, bias=False),
            nn.BatchNorm1d(self.width),
        )
        self.temp_layer = nn.Sequential(
            nn.Linear(1, self.width, bias=False),
            nn.ReLU()
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.temp_layer(beta)
        out = out + self.layers(out)
        if self.include_sigmoid_activation:
            out = torch.sigmoid(out)
        return out
