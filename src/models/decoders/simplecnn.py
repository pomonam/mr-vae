from torch import nn
from src.utils import load_activation
from typing import Tuple


class SimpleCNNDecoder(nn.Module):
    def __init__(self,
                 structure: tuple = (100, 32 * 4, 32 * 2, 32, 256),
                 bias: bool = True):
        super().__init__()

        self.structure = structure
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(self.structure[0], self.structure[1], kernel_size=4, stride=1, padding=0, bias=bias),
            # nn.BatchNorm2d(self.structure[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(self.structure[1], self.structure[2], kernel_size=4, stride=2, padding=1, bias=bias),
            # nn.BatchNorm2d(self.structure[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(self.structure[2], self.structure[3], kernel_size=4, stride=2, padding=1, bias=bias),
            # nn.BatchNorm2d(self.structure[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(self.structure[3], self.structure[4], kernel_size=4, stride=2, padding=1, bias=bias),
        ])

    def forward(self, x, z, *argv):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        for i, layer in enumerate(self.layers):
            z = layer(z, *argv)
        return z
