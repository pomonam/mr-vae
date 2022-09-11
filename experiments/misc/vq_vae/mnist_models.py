from collections import OrderedDict

import torch
from torch import nn

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.models.resblock import ResBlock, HyperResBlock
from src.hyper.base_architecture import BaseHyperDecoder, BaseHyperEncoder
from src.hyper.layers import get_hyper_layer
from src.hyper.norm_layers import get_hyper_bn_layer


class VQMNISTResNetEncoder(BaseEncoder):

  def __init__(self):
    BaseEncoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 64, 4, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
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

    self.pre_qantized = nn.Conv2d(128, self.latent_dim, 1, 1)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        output["embedding"] = self.pre_qantized(out)

    return output


class VQMNISTResNetDecoder(BaseDecoder):

  def __init__(self):
    BaseDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(nn.ConvTranspose2d(self.latent_dim, 128, 1, 1))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
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
            nn.BatchNorm2d(64),
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

      if i + 1 == self.depth:
        output["reconstruction"] = out

    return output


class HyperVQMNISTResNetEncoder(BaseHyperEncoder):

  def __init__(self, hyper_cfg):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1
    self.hyper_cfg = hyper_cfg

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 64, 4, 2, padding=1),
            get_hyper_bn_layer(64, hyper_cfg),
            nn.ReLU(),
            get_hyper_layer(64, hyper_cfg),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, padding=1),
            get_hyper_bn_layer(128, hyper_cfg),
            nn.ReLU(),
            get_hyper_layer(128, hyper_cfg),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, padding=1),
            get_hyper_bn_layer(128, hyper_cfg),
            nn.ReLU(),
            get_hyper_layer(128, hyper_cfg),
        ))

    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            get_hyper_layer(128, hyper_cfg),
            nn.ReLU(),
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            nn.ReLU(),
            get_hyper_layer(128, hyper_cfg),
        ))
    self.layers = layers
    self.depth = len(layers)

    self.pre_qantized = nn.Conv2d(128, self.latent_dim, 1, 1)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        output["embedding"] = self.pre_qantized(out)

    return output


class HyperVQMNISTResNetDecoder(BaseHyperDecoder):

  def __init__(self, hyper_cfg):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(nn.ConvTranspose2d(self.latent_dim, 128, 1, 1))
    # layers.append(get_hyper_layer(128, hyper_cfg))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            get_hyper_bn_layer(128, hyper_cfg),
            nn.ReLU(),
            get_hyper_layer(128, hyper_cfg),
        ))

    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            get_hyper_layer(128, hyper_cfg),
            nn.ReLU(),
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            nn.ReLU(),
            get_hyper_layer(128, hyper_cfg),
        ))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            get_hyper_bn_layer(64, hyper_cfg),
            nn.ReLU(),
            get_hyper_layer(64, hyper_cfg),
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

      if i + 1 == self.depth:
        output["reconstruction"] = out

    return output
