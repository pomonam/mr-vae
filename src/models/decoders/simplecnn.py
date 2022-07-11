from torch import nn
from src.utils import load_activation
from typing import Tuple


class SimpleCNNDecoder(nn.Module):
    def __init__(self,
                 activation: str = "relu",
                 bias: bool = True):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(1, 1, kernel_size=1, stride=7),
            nn.ConvTranspose2d(1, 64, kernel_size=7, stride=1),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.Conv2d(32, 1, kernel_size=5, stride=1),
        ])
        self.activation_fn = load_activation(name=activation)

    def forward(self, x, z, *argv):
        z = z.view(-1, self.structure[0])
        for i, layer in enumerate(self.layers):
            z = layer(z, *argv)
            if i != len(self.layers) - 1:
                z = self.activation_fn(z)
        return z
