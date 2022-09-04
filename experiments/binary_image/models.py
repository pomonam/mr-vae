# Models adapted from:
# https://github.com/clementchadebec/benchmark_VAE/tree/main/src/pythae/models/nn/benchmarks/mnist

from collections import OrderedDict

import torch
from torch import nn

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.models.resblock import ResBlock


class ConvEncoder(BaseEncoder):

  def __init__(self):
    super().__init__()

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()))
    layers.append(
        nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()))
    layers.append(
        nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()))
    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(1024, self.latent_dim)
    self.log_var = nn.Linear(1024, self.latent_dim)

  def forward(self, inputs: torch.Tensor):
    max_depth = self.depth
    out = inputs

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

    return output


class ConvDecoder(BaseDecoder):

  def __init__(self):
    super().__init__()

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(nn.Linear(self.latent_dim, 1024 * 4 * 4))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(
                256, self.n_channels, 3, 2, padding=1, output_padding=1),
            nn.Sigmoid(),
        ))
    self.layers = layers
    self.depth = len(layers)

  def forward(self, inputs: torch.Tensor):
    output = OrderedDict()

    max_depth = self.depth

    out = inputs

    for i in range(max_depth):
      out = self.layers[i](out)

      if i == 0:
        out = out.reshape(inputs.shape[0], 1024, 4, 4)

      if i + 1 == self.depth:
        output["reconstruction"] = out

    return output


class ResNetEncoder(BaseEncoder):

  def __init__(self):
    super().__init__()

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 64, 4, 2, padding=1),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, padding=1),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            ResBlock(channels=128),
            nn.ReLU(),
            ResBlock(channels=128),
            nn.ReLU(),
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
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

    return output


class ResNetDecoder(BaseDecoder):

  def __init__(self):
    super().__init__()

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(nn.Linear(self.latent_dim, 128 * 4 * 4))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            ResBlock(channels=128),
            nn.ReLU(),
            ResBlock(channels=128),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(
                64, self.n_channels, 3, 2, padding=1, output_padding=1),
            nn.Sigmoid()))
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
