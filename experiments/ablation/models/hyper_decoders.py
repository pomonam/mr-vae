from collections import OrderedDict

import torch
from torch import nn

from src.hyper.base_architecture import BaseHyperDecoder
from src.hyper.base_architecture import BaseHyperEncoder
from src.hyper.layers import get_hyper_layer
from src.hyper.norm_layers import get_hyper_bn_layer
from src.models.resblock import HyperResBlock, ResBlock


class HyperResNetEncoder(BaseHyperEncoder):

  def __init__(self, hyper_cfg):
    BaseHyperEncoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1
    self.hyper_cfg = hyper_cfg

    layers = nn.ModuleList()

    if self.hyper_cfg.param_type in ["pre_bn", "post_bn"]:
      layers.append(
          nn.Sequential(
              nn.Conv2d(self.n_channels, 64, 4, 2, padding=1),
              get_hyper_layer(64, hyper_cfg),
              nn.ReLU(),
          ))

      layers.append(
          nn.Sequential(
              nn.Conv2d(64, 128, 4, 2, padding=1),
              get_hyper_layer(128, hyper_cfg),
              nn.ReLU(),
          ))

      layers.append(
          nn.Sequential(
              nn.Conv2d(128, 128, 3, 2, padding=1),
              get_hyper_layer(128, hyper_cfg),
              nn.ReLU(),
          ))

      layers.append(
          nn.Sequential(
              HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
              nn.ReLU(),
              HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
              nn.ReLU(),
          ))

    elif self.hyper_cfg.param_type == "post_act":
      layers.append(
          nn.Sequential(
              nn.Conv2d(self.n_channels, 64, 4, 2, padding=1),
              nn.ReLU(),
              get_hyper_layer(64, hyper_cfg),
          ))

      layers.append(
          nn.Sequential(
              nn.Conv2d(64, 128, 4, 2, padding=1),
              nn.ReLU(),
              get_hyper_layer(128, hyper_cfg),
          ))

      layers.append(
          nn.Sequential(
              nn.Conv2d(128, 128, 3, 2, padding=1),
              nn.ReLU(),
              get_hyper_layer(128, hyper_cfg),
          ))

      layers.append(
          nn.Sequential(
              HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
              nn.ReLU(),
              get_hyper_layer(128, hyper_cfg),
              HyperResBlock(channels=128, hyper_cfg=hyper_cfg),
              nn.ReLU(),
              get_hyper_layer(128, hyper_cfg),
          ))

    else:
      raise NotImplementedError

    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(128 * 4 * 4, 128 * 4 * 4)
    self.hyper_embedding = get_hyper_layer(128 * 4 * 4, hyper_cfg)
    self.embedding_proj = nn.Linear(128 * 4 * 4, self.latent_dim)

    self.log_var = nn.Linear(128 * 4 * 4, 128 * 4 * 4)
    self.hyper_log_var = get_hyper_layer(128 * 4 * 4, hyper_cfg)
    self.log_var_proj = nn.Linear(128 * 4 * 4, self.latent_dim)

  def forward(self, inputs: torch.Tensor):
    max_depth = self.depth
    out = inputs

    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        if self.hyper_cfg.include_latent_stem:
          emb = self.embedding(out.reshape(inputs.shape[0], -1))
          emb = self.hyper_embedding(emb)
          emb = self.embedding_proj(emb)
          output["embedding"] = emb
          lv = self.log_var(out.reshape(inputs.shape[0], -1))
          lv = self.hyper_log_var(lv)
          lv = self.log_var_proj(lv)
          output["log_covariance"] = lv
        else:
          emb = self.embedding_proj(out.reshape(inputs.shape[0], -1))
          output["embedding"] = emb
          lv = self.log_var_proj(out.reshape(inputs.shape[0], -1))
          output["log_covariance"] = lv
    return output


class HyperResNetDecoder(BaseHyperDecoder):

  def __init__(self, hyper_cfg):
    BaseHyperDecoder.__init__(self)

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32
    self.n_channels = 1
    self.hyper_cfg = hyper_cfg

    layers = nn.ModuleList()

    layers.append(nn.Linear(self.latent_dim, 128 * 4 * 4))
    # layers.append(get_hyper_layer(128 * 4 * 4, hyper_cfg))

    layers.append(
        nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, padding=1),
            # get_hyper_layer(128, hyper_cfg),
            nn.ReLU(),
        ))
    layers.append(
        nn.Sequential(
            ResBlock(channels=128, hyper_cfg=hyper_cfg),
            nn.ReLU(),
            ResBlock(channels=128, hyper_cfg=hyper_cfg),
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
