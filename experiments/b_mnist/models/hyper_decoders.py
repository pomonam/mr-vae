import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from src.hyper.layers.conv2d import HyperConv2d, HyperConvTranspose2d
from src.hyper.layers.linear import HyperLinear
from src.hyper.models import BaseHyperDecoder
from src.models.base_decoder import BaseDecoder
from src.models.pixcelcnn import PixelCNN


class HyperMLPDecoder(BaseHyperDecoder):

    def __init__(self, hyper_config):
        super().__init__()

        self.hyper_config = hyper_config
        self.linear1 = HyperLinear(64, 256, "relu", hyper_config)
        self.linear2 = HyperLinear(256, 512, "relu", hyper_config)
        self.linear3 = HyperLinear(512, 512, "relu", hyper_config)
        self.linear4 = HyperLinear(512, 784, "none", hyper_config)

    def forward(self, z):
        z = self.linear1(z)
        z = self.linear2(z)
        z = self.linear3(z)
        z = self.linear4(z)
        z = z.view(z.shape[0], 1, 28, 28)
        return z


class HyperCNNDecoder(BaseHyperDecoder):

    def __init__(self, hyper_config):
        super().__init__()

        self.hyper_config = hyper_config
        self.initial_layer = HyperLinear(64, 32 * 8, "none", hyper_config)

        self.layers = nn.Sequential(
            HyperConvTranspose2d(
                32 * 8,
                32 * 4,
                activation_fnc="relu",
                hyper_config=hyper_config,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1),
            HyperConvTranspose2d(
                32 * 4,
                32 * 2,
                activation_fnc="relu",
                hyper_config=hyper_config,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=1),
            HyperConvTranspose2d(
                32 * 2, 32,  activation_fnc="relu",
                hyper_config=hyper_config, kernel_size=4, stride=2, padding=1),
            HyperConvTranspose2d(32, 1, activation_fnc="none", hyper_config=hyper_config, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        z = self.initial_layer(z)
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        z = self.layers(z)
        return z


class PixelCNNDecoder(BaseDecoder):
    require_inputs = True

    def __init__(self):
        super(PixelCNNDecoder, self).__init__()
        self.nz = 64
        self.nc = 1
        self.fm_latent = 4

        self.img_latent = 28 * 28 * self.fm_latent
        if self.nz != 0:
            self.z_transform = nn.Sequential(
                nn.Linear(self.nz, self.img_latent),)
        kernal_sizes = [9, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3, 3]

        hidden_channels = 32
        self.layers = nn.Sequential(
            PixelCNN(1 + self.fm_latent,
                     hidden_channels,
                     len(kernal_sizes),
                     kernal_sizes,
                     self.nc),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, self.nc, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise ValueError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise ValueError

    def special_decode(self, z, x):
        z = self.z_transform(z)
        z = z.view(-1, self.fm_latent, 28, 28)
        z = torch.cat([x, z], dim=1)
        return self.layers(z)

    def special_forward(self, z, x):
        return self.special_decode(z, x)
