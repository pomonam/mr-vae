from torch import nn
from src.utils import load_activation
from typing import Tuple
from src.models.encoders.base import BaseEncoder
from src.utils import get_conv_layers


class SimpleCNNEncoder(BaseEncoder):
    def __init__(self,
                 structure: list = [3, 32, 64, 128, 64],
                 bias: bool = True):
        super().__init__()

        assert len(structure) == 5
        self.structure = structure

        self.layers = nn.ModuleList([
            nn.Conv2d(self.structure[0], self.structure[1], kernel_size=4, stride=2, padding=1, bias=bias),
            # nn.BatchNorm2d(self.structure[1]),
            nn.ReLU(),
            nn.Conv2d(self.structure[1], self.structure[2], kernel_size=4, stride=2, padding=1, bias=bias),
            # nn.BatchNorm2d(self.structure[2]),
            nn.ReLU(),
            nn.Conv2d(self.structure[2], self.structure[3], kernel_size=4, stride=2, padding=1, bias=bias),
            # nn.BatchNorm2d(self.structure[3]),
            nn.ReLU(),
            nn.Conv2d(self.structure[3], self.structure[4], kernel_size=4, stride=1, padding=1, bias=bias),
            nn.Flatten()
        ])

    def forward(self, x, *argv):
        for i, layer in enumerate(self.layers):
            x = layer(x, *argv)
        return x
