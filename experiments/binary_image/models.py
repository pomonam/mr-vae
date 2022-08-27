from collections import OrderedDict

import torch
import torch.nn as nn

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.models.resblock import ResBlock


class ResNetEncoder(BaseEncoder):

    def __init__(self):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = 32
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(
            nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))
        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))
        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))
        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            ))

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, self.latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, self.latent_dim)

    def forward(self, x: torch.Tensor):
        max_depth = self.depth
        out = x

        output = {}
        for i in range(max_depth):
            out = self.layers[i](out)

            if i + 1 == self.depth:
                output["embedding"] = self.embedding(
                    out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(
                    out.reshape(x.shape[0], -1))

        return output


class ResNetDecoder(BaseDecoder):

    def __init__(self):
        BaseDecoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = 32
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Linear(self.latent_dim, 128 * 4 * 4))
        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))
        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
                nn.ReLU(),
            ))
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
            ))
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    64, self.n_channels, 3, 2, padding=1, output_padding=1),
                nn.Sigmoid(),
            ))

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor):
        output = OrderedDict()

        max_depth = self.depth

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 128, 4, 4)

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output
