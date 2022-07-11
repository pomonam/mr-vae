from torch import nn
from src.utils import load_activation
from typing import Tuple
from src.models.decoders.base import BaseDecoder


class MLPDecoder(BaseDecoder):
    def __init__(self,
                 structure: Tuple[int, ...] = (784, 70, 10),
                 activation: str = "relu",
                 bias: bool = True):
        super().__init__()
        self.structure = structure
        self.layers = nn.ModuleList([
            nn.Linear(a, b, bias=bias)
            for a, b in zip(structure, structure[1:])
        ])
        self.activation_fn = load_activation(name=activation)

    def forward(self, x, z, *argv):
        z = z.view(-1, self.structure[0])
        for i, layer in enumerate(self.layers):
            z = layer(z, *argv)
            if i != len(self.layers) - 1:
                z = self.activation_fn(z)
        return z
