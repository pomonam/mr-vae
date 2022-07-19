import torch
from torch import nn


class BaseBlock(nn.Module):

    def __init__(self,
                 width: int,
                 include_output_linear: bool = True,
                 include_sigmoid_activation: bool = True):
        super().__init__()
        self.width = width
        self.include_output_linear = include_output_linear
        self.include_sigmoid_activation = include_sigmoid_activation

        if self.include_output_linear:
            self.output_layer = nn.Linear(self.width, self.width, bias=True)
        else:
            self.output_layer = None

        self.layers = None
        self._beta = None
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
        self.layers[0].weight.data.fill_(0)
        self.layers[0].bias.data.fill_(0)

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.layers(beta)
        if self.include_sigmoid_activation:
            out = torch.sigmoid(out)
        if self.include_output_linear:
            out = self.output_layer(out)
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
        if self.include_output_linear:
            out = self.output_layer(out)
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
            nn.Linear(1, self.width),
            nn.ReLU()
        )
        self.layers[-1].weight.data.fill_(0)

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.temp_layer(beta)
        out = out + self.layers(out)
        if self.include_sigmoid_activation:
            out = torch.sigmoid(out)
        if self.include_output_linear:
            out = self.output_layer(out)
        return out


class BatchNormResidualBlock(nn.Module):

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
            nn.Linear(1, self.width),
            nn.ReLU()
        )

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        out = self.temp_layer(beta)
        out = out + self.layers(out)
        if self.include_sigmoid_activation:
            out = torch.sigmoid(out)
        if self.include_output_linear:
            out = self.output_layer(out)
        return out
