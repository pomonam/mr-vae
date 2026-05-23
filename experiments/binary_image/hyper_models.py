from collections import OrderedDict

import torch
from torch import nn

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.layers import get_hyper_layer
from src.models.resblock import HyperResBlock


class HyperConvEncoder(BaseHyperEncoder):

  def __init__(self):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            get_hyper_layer(128),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            get_hyper_layer(256),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            get_hyper_layer(512),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            get_hyper_layer(1024),
            nn.ReLU(),
        ))
    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(1024, self.latent_dim)
    self.log_var = nn.Linear(1024, self.latent_dim)

  def forward(self, x: torch.Tensor):
    out = x
    output = {}
    for i in range(self.depth):
      out = self.layers[i](out)
      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))
    return output


class HyperConvDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(nn.Linear(self.latent_dim, 1024 * 4 * 4))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, padding=1),
            nn.BatchNorm2d(512),
            get_hyper_layer(512, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            get_hyper_layer(256, decoder=True),
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
    out = inputs
    for i in range(self.depth):
      out = self.layers[i](out)
      if i == 0:
        out = out.reshape(inputs.shape[0], 1024, 4, 4)
      if i + 1 == self.depth:
        output["reconstruction"] = out
    return output


class HyperResNetEncoder(BaseHyperEncoder):

  def __init__(self):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 64, 4, 2, padding=1),
            get_hyper_layer(64),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, padding=1),
            get_hyper_layer(128),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, padding=1),
            get_hyper_layer(128),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128),
            nn.ReLU(),
            HyperResBlock(channels=128),
            nn.ReLU(),
        ))
    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(128 * 4 * 4, self.latent_dim)
    self.log_var = nn.Linear(128 * 4 * 4, self.latent_dim)

  def forward(self, inputs: torch.Tensor):
    out = inputs
    output = {}
    for i in range(self.depth):
      out = self.layers[i](out)
      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out.reshape(inputs.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(inputs.shape[0], -1))
    return output


class HyperResNetDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(nn.Linear(self.latent_dim, 128 * 4 * 4))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            get_hyper_layer(128, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, decoder=True),
            nn.ReLU(),
            HyperResBlock(channels=128, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            get_hyper_layer(64, decoder=True),
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
    out = z
    for i in range(self.depth):
      out = self.layers[i](out)
      if i == 0:
        out = out.reshape(z.shape[0], 128, 4, 4)
      if i + 1 == self.depth:
        output["reconstruction"] = out
    return output
