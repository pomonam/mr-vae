from collections import OrderedDict

import torch
import torch.nn as nn

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.layer import HyperLayer
from src.models.resblock import ResBlock


class HyperResNetCifarEncoder(BaseHyperEncoder):

  def __init__(self, hyper_cfg):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = 32
    self.n_channels = 3

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))
    layers.append(HyperLayer(64, hyper_cfg))
    layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))
    layers.append(HyperLayer(128, hyper_cfg))
    layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, padding=1)))
    layers.append(HyperLayer(128, hyper_cfg))
    layers.append(
        nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
            HyperLayer(128, hyper_cfg),
            ResBlock(in_channels=128, out_channels=32),
            HyperLayer(128, hyper_cfg),
        ))

    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(128 * 8 * 8, self.latent_dim)
    self.hyper_embedding = HyperLayer(
        self.latent_dim, hyper_cfg, use_group=False)
    self.embedding_proj = nn.Linear(self.latent_dim, self.latent_dim)

    self.log_var = nn.Linear(128 * 8 * 8, self.latent_dim)
    self.hyper_log_var = HyperLayer(self.latent_dim, hyper_cfg, use_group=False)
    self.log_var_proj = nn.Linear(self.latent_dim, self.latent_dim)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        emb = self.embedding(out.reshape(x.shape[0], -1))
        emb = self.hyper_embedding(emb)
        emb = self.embedding_proj(emb)
        output["embedding"] = emb
        lv = self.log_var(out.reshape(x.shape[0], -1))
        lv = self.hyper_log_var(lv)
        lv = self.log_var_proj(lv)
        output["log_covariance"] = lv

    return output


class HyperResNetCifarDecoder(BaseHyperDecoder):

  def __init__(self, hyper_cfg):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 32, 32)
    self.latent_dim = 32
    self.n_channels = 3

    layers = nn.ModuleList()

    layers.append(nn.Linear(self.latent_dim, 128 * 8 * 8))
    layers.append(HyperLayer(128 * 8 * 8, hyper_cfg, use_group=False))

    layers.append(
        nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
            HyperLayer(128, hyper_cfg),
            ResBlock(in_channels=128, out_channels=32),
            HyperLayer(128, hyper_cfg),
        ))
    layers.append(nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, padding=1)))
    layers.append(HyperLayer(64, hyper_cfg))
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

      if i == 1:
        out = out.reshape(z.shape[0], 128, 8, 8)

      if i + 1 == self.depth:
        output["reconstruction"] = out

    return output


class HyperResNetCelebEncoder(BaseHyperEncoder):

  def __init__(self, hyper_cfg):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))
    layers.append(HyperLayer(64, hyper_cfg))

    layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))
    layers.append(HyperLayer(128, hyper_cfg))
    layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))
    layers.append(HyperLayer(128, hyper_cfg))
    layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))
    layers.append(HyperLayer(128, hyper_cfg))

    layers.append(
        nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
          HyperLayer(128, hyper_cfg),

          ResBlock(in_channels=128, out_channels=32),
          HyperLayer(128, hyper_cfg),

        ))

    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(128 * 4 * 4, self.latent_dim)
    self.hyper_embedding = HyperLayer(
        self.latent_dim, hyper_cfg, use_group=False)
    self.embedding_proj = nn.Linear(self.latent_dim, self.latent_dim)

    self.log_var = nn.Linear(128 * 4 * 4, self.latent_dim)
    self.hyper_log_var = HyperLayer(self.latent_dim, hyper_cfg, use_group=False)
    self.log_var_proj = nn.Linear(self.latent_dim, self.latent_dim)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        emb = self.embedding(out.reshape(x.shape[0], -1))
        emb = self.hyper_embedding(emb)
        emb = self.embedding_proj(emb)
        output["embedding"] = emb
        lv = self.log_var(out.reshape(x.shape[0], -1))
        lv = self.hyper_log_var(lv)
        lv = self.log_var_proj(lv)
        output["log_covariance"] = lv

    return output


class HyperResNetCelebDecoder(BaseHyperDecoder):

  def __init__(self, hyper_cfg):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (3, 64, 64)
    self.latent_dim = 64
    self.n_channels = 3

    layers = nn.ModuleList()

    layers.append(nn.Linear(self.latent_dim, 128 * 4 * 4))
    layers.append(HyperLayer(128 * 4 * 4, hyper_cfg, use_group=False))

    layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))
    layers.append(HyperLayer(128, hyper_cfg))

    layers.append(
        nn.Sequential(
            ResBlock(in_channels=128, out_channels=32),
            HyperLayer(128, hyper_cfg),
            ResBlock(in_channels=128, out_channels=32),
            HyperLayer(128, hyper_cfg),
        ))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 128, 5, 2, padding=1), HyperLayer(128, hyper_cfg), nn.Sigmoid()))
    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5, 2, padding=1, output_padding=1), HyperLayer(64, hyper_cfg)))
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

      if i == 1:
        out = out.reshape(z.shape[0], 128, 4, 4)

      if i + 1 == self.depth:
        output["reconstruction"] = out

    return output
