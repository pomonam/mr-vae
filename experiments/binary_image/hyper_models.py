from collections import OrderedDict

import torch
import torch.nn as nn

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.layer import HyperLayer
from src.models.resblock import HyperResBlock


class HyperConvEncoder(BaseHyperEncoder):

  def __init__(self, hyper_cfg):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1
    self.hyper_cfg = hyper_cfg

    layers = nn.ModuleList()
    layers.append(
        nn.Sequential(
            nn.Conv2d(self.n_channels, 128, 4, 2, padding=1),
            HyperLayer(128, hyper_cfg),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, padding=1),
            HyperLayer(256, hyper_cfg),
            nn.BatchNorm2d(256),
            nn.ReLU()))
    layers.append(
        nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1),
            HyperLayer(512, hyper_cfg),
            nn.BatchNorm2d(512),
            nn.ReLU()))
    layers.append(
        nn.Sequential(
            nn.Conv2d(512, 1024, 4, 2, padding=1),
            HyperLayer(1024, hyper_cfg),
            nn.BatchNorm2d(1024),
            nn.ReLU()))
    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(1024, self.latent_dim)
    self.hyper_embedding = HyperLayer(
        self.latent_dim, hyper_cfg, use_group=False)
    self.embedding_proj = nn.Linear(self.latent_dim, self.latent_dim)

    self.log_var = nn.Linear(1024, self.latent_dim)
    self.hyper_log_var = HyperLayer(self.latent_dim, hyper_cfg, use_group=False)
    self.log_var_proj = nn.Linear(self.latent_dim, self.latent_dim)

  def forward(self, x: torch.Tensor):
    max_depth = self.depth
    out = x

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        if self.hyper_cfg.include_output_stem:
          emb = self.embedding(out.reshape(x.shape[0], -1))
          emb = self.hyper_embedding(emb)
          emb = self.embedding_proj(emb)
          output["embedding"] = emb

          lv = self.log_var(out.reshape(x.shape[0], -1))
          lv = self.hyper_log_var(lv)
          lv = self.log_var_proj(lv)
          output["log_covariance"] = lv

        else:
          output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
          output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

    return output


class HyperConvDecoder(BaseHyperDecoder):

  def __init__(self, hyper_cfg):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(nn.Linear(self.latent_dim, 1024 * 4 * 4))
    layers.append(HyperLayer(1024 * 4 * 4, hyper_cfg, use_group=False))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, 2, padding=1),
            HyperLayer(512, hyper_cfg),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1),
            HyperLayer(256, hyper_cfg),
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

  def forward(self, z: torch.Tensor):
    output = OrderedDict()

    max_depth = self.depth

    out = z

    for i in range(max_depth):
      out = self.layers[i](out)

      if i == 1:
        out = out.reshape(z.shape[0], 1024, 4, 4)

      if i + 1 == self.depth:
        output["reconstruction"] = out

    return output


class HyperResNetEncoder(BaseHyperEncoder):

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
            HyperLayer(64, hyper_cfg),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, padding=1),
            HyperLayer(128, hyper_cfg),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, padding=1),
            HyperLayer(128, hyper_cfg),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            nn.ReLU(),
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            nn.ReLU(),
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
        if self.hyper_cfg.include_output_stem:
          emb = self.embedding(out.reshape(x.shape[0], -1))
          emb = self.hyper_embedding(emb)
          emb = self.embedding_proj(emb)
          output["embedding"] = emb
          lv = self.log_var(out.reshape(x.shape[0], -1))
          lv = self.hyper_log_var(lv)
          lv = self.log_var_proj(lv)
          output["log_covariance"] = lv
        else:
          emb = self.embedding(out.reshape(x.shape[0], -1))
          output["embedding"] = emb
          lv = self.log_var(out.reshape(x.shape[0], -1))
          output["log_covariance"] = lv
    return output


class HyperResNetDecoder(BaseHyperDecoder):

  def __init__(self, hyper_cfg):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1

    layers = nn.ModuleList()

    layers.append(nn.Linear(self.latent_dim, 128 * 4 * 4))
    layers.append(HyperLayer(128 * 4 * 4, hyper_cfg, use_group=False))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            HyperLayer(128, hyper_cfg),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            nn.ReLU(),
            HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
            nn.ReLU(),
        ))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            HyperLayer(64, hyper_cfg),
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

      if i == 1:
        out = out.reshape(z.shape[0], 128, 4, 4)

      if i + 1 == self.depth:
        output["reconstruction"] = out

    return output
