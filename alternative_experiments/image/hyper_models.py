from collections import OrderedDict

import torch
import torch.nn as nn

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.modules import HyperLinear, HyperConv2d, HyperConvTranspose2d


class CustomReLU(nn.Module):
  def forward(self, x):
    return torch.clamp(x, min=0)


class HyperResBlock(nn.Module):

  def __init__(self, channels: int, decoder: bool = False) -> None:
    super().__init__()

    self.conv_block = nn.Sequential(
        HyperConv2d(channels, channels, kernel_size=3, stride=1, padding=1, decoder=decoder),
        CustomReLU(),
        HyperConv2d(channels, channels, kernel_size=3, stride=1, padding=1, decoder=decoder),
      )

  def forward(self, x: torch.tensor) -> torch.Tensor:
    return x + self.conv_block(x)


class HyperCifarConvEncoder(BaseHyperEncoder):

  def __init__(self, latent_dim):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = latent_dim
    self.n_channels = 3
    layers = nn.ModuleList()

    layers.append(
        nn.Sequential(
            HyperConv2d(self.n_channels, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            CustomReLU()))

    self.layers = layers
    self.depth = len(layers)

    self.embedding = HyperLinear(1024 * 2 * 2, self.latent_dim)
    self.log_var = HyperLinear(1024 * 2 * 2, self.latent_dim)

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


class HyperCifarConvDecoder(BaseHyperDecoder):

  def __init__(self, latent_dim):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = latent_dim
    self.n_channels = 3

    layers = nn.ModuleList()

    layers.append(HyperLinear(self.latent_dim, 1024 * 8 * 8, decoder=True))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(1024, 512, 4, 2, padding=1, decoder=True),
            nn.BatchNorm2d(512),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(512, 256, 4, 2, padding=1, output_padding=1, decoder=True),
            nn.BatchNorm2d(512),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(256, self.n_channels, 4, 1, padding=2),
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
            HyperConv2d(self.n_channels, 64, 4, 2, padding=1),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConv2d(64, 128, 4, 2, padding=1),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConv2d(128, 128, 3, 1, padding=1),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperResBlock(128),
            CustomReLU(),
            HyperResBlock(128),
            CustomReLU(),
        ))

    self.layers = layers
    self.depth = len(layers)

    self.embedding = HyperLinear(128 * 8 * 8, self.latent_dim)
    self.log_var = HyperLinear(128 * 8 * 8, self.latent_dim)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        emb = self.embedding(out.reshape(x.shape[0], -1))
        output["embedding"] = emb
        lv = self.log_var(out.reshape(x.shape[0], -1))
        output["log_covariance"] = lv

    return output


class HyperCifarResNetDecoder(BaseHyperDecoder):

  def __init__(self, latent_dim):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = latent_dim
    self.n_channels = 3

    layers = nn.ModuleList()

    layers.append(HyperLinear(self.latent_dim, 128 * 8 * 8, decoder=True))

    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, decoder=True),
            CustomReLU(),
            HyperResBlock(channels=128, decoder=True),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(128, 64, 4, 2, padding=1, decoder=True),
            CustomReLU()))

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
            HyperConv2d(self.n_channels, 128, 4, 2, padding=1),
            nn.BatchNorm2d(128),
            CustomReLU(),
        ))
    layers.append(
        nn.Sequential(
            HyperConv2d(128, 256, 4, 2, padding=1),
            nn.BatchNorm2d(256),
            CustomReLU(),
        ))

    layers.append(
        nn.Sequential(
            HyperConv2d(256, 512, 4, 2, padding=1),
            nn.BatchNorm2d(512),
            CustomReLU(),
        ))
    layers.append(
        nn.Sequential(
            HyperConv2d(512, 1024, 4, 2, padding=1),
            nn.BatchNorm2d(1024),
            CustomReLU(),
        ))
    self.layers = layers
    self.depth = len(layers)

    self.embedding = HyperLinear(1024 * 4 * 4, self.latent_dim)
    self.log_var = HyperLinear(1024 * 4 * 4, self.latent_dim)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        emb = self.embedding(out.reshape(x.shape[0], -1))
        output["embedding"] = emb
        lv = self.log_var(out.reshape(x.shape[0], -1))
        output["log_covariance"] = lv
    return output


class HyperCelebConvDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

    layers = nn.ModuleList()

    layers.append(HyperLinear(self.latent_dim, 1024 * 8 * 8, decoder=True))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(1024, 512, 5, 2, padding=2, decoder=True),
            nn.BatchNorm2d(256),
            CustomReLU(),))
    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(512, 256, 5, 2, padding=1, output_padding=0, decoder=True),
            nn.BatchNorm2d(256),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(256, 128, 5, 2, padding=2, output_padding=1, decoder=True),
            nn.BatchNorm2d(256),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, self.n_channels, 5, 1, padding=1),
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
            HyperConv2d(self.n_channels, 64, 4, 2, padding=1),
            CustomReLU()))
    layers.append(
        nn.Sequential(
            HyperConv2d(64, 128, 4, 2, padding=1),
            CustomReLU(),))
    layers.append(
        nn.Sequential(
            HyperConv2d(128, 128, 3, 2, padding=1),
            CustomReLU()))
    layers.append(
        nn.Sequential(
            HyperConv2d(128, 128, 3, 2, padding=1),
            CustomReLU()))

    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128),
            CustomReLU(),
            HyperResBlock(channels=128),
            CustomReLU(),))

    self.layers = layers
    self.depth = len(layers)

    self.embedding = HyperLinear(128 * 4 * 4, self.latent_dim)
    self.log_var = HyperLinear(128 * 4 * 4, self.latent_dim)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        emb = self.embedding(out.reshape(x.shape[0], -1))
        output["embedding"] = emb
        lv = self.log_var(out.reshape(x.shape[0], -1))
        output["log_covariance"] = lv
    return output


class HyperCelebResNetDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

    layers = nn.ModuleList()

    layers.append(HyperLinear(self.latent_dim, 128 * 4 * 4, decoder=True))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(128, 128, 3, 2, padding=1),
            CustomReLU(),
        ))
    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, decoder=True),
            CustomReLU(),
            HyperResBlock(channels=128, decoder=True),
            CustomReLU(),
        ))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(
                128,
                128,
                5,
                2,
                padding=1,
              decoder=True
            ),
            nn.Sigmoid(),
        ))
    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(128, 64, 5, 2, padding=1, output_padding=1, decoder=True),
            CustomReLU(),
        ))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(64, self.n_channels, 4, 2, padding=1),
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
