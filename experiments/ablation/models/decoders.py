from collections import OrderedDict
import math
import torch
from torch import nn
import texar.torch as tx
from src.models.pixelcnn import PixelCNN
from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.models.resblock import ResBlock


class PixelCnnDecoder(BaseDecoder):

  def __init__(self):
    super().__init__()

    self.latent_size = 32
    self.num_channels = 1
    self.fm_latent = 4
    self.img_latent = 28 * 28 * self.fm_latent

    self.z_transform = nn.Sequential(
      nn.Linear(self.latent_size, self.img_latent),
    )

    kernal_sizes = [7, 7, 7, 5, 5, 3, 3]
    hidden_channels = 64
    self.layers = nn.Sequential(
      PixelCNN(self.num_channels + self.fm_latent, hidden_channels, len(kernal_sizes), kernal_sizes, self.latent_size),
      nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
      nn.BatchNorm2d(hidden_channels),
      nn.ELU(),
      nn.Conv2d(hidden_channels, self.num_channels, 1, bias=False),
      nn.Sigmoid(),
    )

  def forward(self, inputs):
    raise LookupError

  def ar_forward(self, x, z, inputs):
    z = z.unsqueeze(1)
    batch_size, nsampels, nz = z.size()
    z = self.z_transform(z).view(batch_size, nsampels, self.fm_latent, 28, 28)
    img = x.unsqueeze(1).expand(batch_size, nsampels, *x.size()[1:])
    img = torch.cat([img, z], dim=2)
    img = img.view(-1, *img.size()[2:])
    recon_x = self.layers(img).view(batch_size, nsampels, -1)
    recon_loss = torch.nn.functional.binary_cross_entropy(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
    ).sum(dim=-1)

    return recon_loss

  def decode(self, z):
    H = W = 28
    batch_size, nz = z.size()

    # [batch, -1] --> [batch, fm, H, W]
    z = self.z_transform(z).view(batch_size, self.fm_latent, H, W)
    img = z.data.new(batch_size, self.num_channels, H, W).zero_()
    # [batch, nc+fm, H, W]
    img = torch.cat([img, z], dim=1)
    for i in range(H):
      for j in range(W):
        # [batch, nc, H, W]
        recon_img = self.layers(img)
        # [batch, nc]
        img[:, :self.num_channels, i, j] = torch.ge(recon_img[:, :, i, j], 0.5).float()
        # img[:, :self.nc, i, j] = torch.bernoulli(recon_img[:, :, i, j])

    # [batch, nc, H, W]
    img_probs = self.layers(img)
    return img_probs
