from collections import OrderedDict

import torch
import torch.nn as nn

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.layers import get_hyper_layer
from src.models.resblock import HyperResBlock


class HyperCifarConvEncoder(BaseHyperEncoder):

  def __init__(self, latent_dim):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = latent_dim
    self.n_channels = 3

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

    self.embedding = nn.Linear(1024 * 2 * 2, self.latent_dim)
    self.log_var = nn.Linear(1024 * 2 * 2, self.latent_dim)

  def forward(self, x: torch.Tensor):
    out = x
    output = {}
    for i in range(self.depth):
      out = self.layers[i](out)
      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))
    return output


class HyperCifarConvDecoder(BaseHyperDecoder):

  def __init__(self, latent_dim):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = latent_dim
    self.n_channels = 3

    layers = nn.ModuleList()
    layers.append(nn.Linear(self.latent_dim, 1024 * 8 * 8))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            get_hyper_layer(512, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            get_hyper_layer(256, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(256, self.n_channels, 4, 1, padding=2),
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
        out = out.reshape(z.shape[0], 1024, 8, 8)
      if i + 1 == self.depth:
        output["reconstruction"] = out
    return output


class HyperCifarResNetEncoder(BaseHyperEncoder):

  def __init__(self, latent_dim):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = latent_dim
    self.n_channels = 3

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
            nn.Conv2d(128, 128, 3, 1, padding=1),
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

    self.embedding = nn.Linear(128 * 8 * 8, self.latent_dim)
    self.log_var = nn.Linear(128 * 8 * 8, self.latent_dim)

  def forward(self, x: torch.Tensor):
    out = x
    output = {}
    for i in range(self.depth):
      out = self.layers[i](out)
      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))
    return output


class HyperCifarResNetDecoder(BaseHyperDecoder):

  def __init__(self, latent_dim):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = latent_dim
    self.n_channels = 3

    layers = nn.ModuleList()
    layers.append(nn.Linear(self.latent_dim, 128 * 8 * 8))
    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, decoder=True),
            nn.ReLU(),
            HyperResBlock(channels=128, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
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
        out = out.reshape(z.shape[0], 128, 8, 8)
      if i + 1 == self.depth:
        output["reconstruction"] = out
    return output


class HyperCelebConvEncoder(BaseHyperEncoder):

  def __init__(self):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

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

    self.embedding = nn.Linear(1024 * 4 * 4, self.latent_dim)
    self.log_var = nn.Linear(1024 * 4 * 4, self.latent_dim)

  def forward(self, x: torch.Tensor):
    out = x
    output = {}
    for i in range(self.depth):
      out = self.layers[i](out)
      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))
    return output


class HyperCelebConvDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

    layers = nn.ModuleList()
    layers.append(nn.Linear(self.latent_dim, 1024 * 8 * 8))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 2, padding=2),
            nn.BatchNorm2d(512),
            get_hyper_layer(512, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            get_hyper_layer(256, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            get_hyper_layer(128, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1),
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
        out = out.reshape(z.shape[0], 1024, 8, 8)
      if i + 1 == self.depth:
        output["reconstruction"] = out
    return output


class HyperCelebResNetEncoder(BaseHyperEncoder):

  def __init__(self):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

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

  def forward(self, x: torch.Tensor):
    out = x
    output = {}
    for i in range(self.depth):
      out = self.layers[i](out)
      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
        output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))
    return output


class HyperCelebResNetDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

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
            nn.ConvTranspose2d(128, 128, 5, 2, padding=1),
            get_hyper_layer(128, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2, padding=1, output_padding=1),
            get_hyper_layer(64, decoder=True),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(64, self.n_channels, 4, 2, padding=1),
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
