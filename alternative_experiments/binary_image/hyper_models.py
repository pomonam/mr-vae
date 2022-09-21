from collections import OrderedDict

import torch
from torch import nn

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


class HyperConvEncoder(BaseHyperEncoder):

  def __init__(self):
    super().__init__()

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            HyperConv2d(self.n_channels, 128, 4, 2, padding=1),
            CustomReLU(),
        ))
    layers.append(
        nn.Sequential(
            HyperConv2d(128, 256, 4, 2, padding=1),
            CustomReLU()))
    layers.append(
        nn.Sequential(
            HyperConv2d(256, 512, 4, 2, padding=1),
            CustomReLU()))
    layers.append(
        nn.Sequential(
            HyperConv2d(512, 1024, 4, 2, padding=1),
            CustomReLU()))

    self.layers = layers
    self.depth = len(layers)

    self.embedding = HyperLinear(1024, self.latent_dim)
    self.log_var = HyperLinear(1024, self.latent_dim)

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


class HyperConvDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(HyperLinear(self.latent_dim, 1024 * 4 * 4, decoder=True))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(1024, 512, 3, 2, padding=1, decoder=True),
            CustomReLU(),
        ))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1, decoder=True),
            CustomReLU(),
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


class HyperResNetEncoder(BaseHyperEncoder):

  def __init__(self):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(
        nn.Sequential(
            HyperConv2d(self.n_channels, 64, 4, 2, padding=1),
            CustomReLU(),
        ))

    layers.append(
        nn.Sequential(
            HyperConv2d(64, 128, 4, 2, padding=1),
            CustomReLU(),
        ))

    layers.append(
        nn.Sequential(
            HyperConv2d(128, 128, 3, 2, padding=1),
            CustomReLU(),
        ))

    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128),
            CustomReLU(),
            HyperResBlock(channels=128),
            CustomReLU(),
        ))

    self.layers = layers
    self.depth = len(layers)

    self.embedding = HyperLinear(128 * 4 * 4, self.latent_dim)
    self.log_var = HyperLinear(128 * 4 * 4, self.latent_dim)

  def forward(self, inputs: torch.Tensor):
    max_depth = self.depth
    out = inputs

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        emb = self.embedding(out.reshape(inputs.shape[0], -1))
        output["embedding"] = emb
        lv = self.log_var(out.reshape(inputs.shape[0], -1))
        output["log_covariance"] = lv
    return output


class HyperResNetDecoder(BaseHyperDecoder):

  def __init__(self):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(HyperLinear(self.latent_dim, 128 * 4 * 4, decoder=True))

    layers.append(
        nn.Sequential(
            HyperConvTranspose2d(128, 128, 3, 2, padding=1, decoder=True),
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
            HyperConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1, decoder=True),
            CustomReLU(),
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
