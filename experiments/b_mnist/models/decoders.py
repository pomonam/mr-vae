import torch
from torch import nn
from torch.autograd import Variable

from src.models.base_decoder import BaseDecoder
from src.models.pixcelcnn import PixelCNN


class MLPDecoder(BaseDecoder):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(64, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, 784)

    def forward(self, z):
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)
        z = torch.relu(z)
        z = self.linear3(z)
        z = torch.relu(z)
        z = self.linear4(z)
        z = z.view(z.shape[0], 1, 28, 28)
        return z


class CNNDecoder(BaseDecoder):
    def __init__(self):
        super().__init__()

        self.initial_layer = nn.Linear(64, 32 * 8)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(32 * 8,
                               32 * 4,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32 * 4,
                               32 * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32 * 2, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
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
                nn.Linear(self.nz, self.img_latent), )
        kernal_sizes = [9, 9, 9, 7, 7, 7, 5, 5, 5, 3, 3, 3]

        hidden_channels = 32
        self.layers = nn.Sequential(
            PixelCNN(1 + self.fm_latent, hidden_channels, len(kernal_sizes),
                     kernal_sizes, self.nc),
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
