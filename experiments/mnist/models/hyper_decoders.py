import torch
from torch import nn
import math

from src.hyper.layers.conv2d import HyperConv2d, HyperConvTranspose2d
from src.hyper.layers.linear import HyperLinear
from src.hyper.models import BaseHyperDecoder
from src.models.base_decoder import BaseDecoder
from src.models.pixcelcnn import PixelCNN, MaskedConv2d, MaskABlock


class HyperMLPDecoder(BaseHyperDecoder):

    def __init__(self, hyper_config):
        super().__init__()

        self.hyper_config = hyper_config
        self.linear1 = HyperLinear(32, 256, "relu", hyper_config)
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
        self.initial_layer = HyperLinear(32, 32 * 8, "none", hyper_config)
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
            HyperConvTranspose2d(32, 1, activation_fnc="none", hyper_config=hyper_config,
                                 kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        z = self.initial_layer(z)
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        z = self.layers(z)
        return z


class HyperPixelCNNBlock(nn.Module):

    def __init__(self, in_channels, kernel_size, hyper_config):
        super(HyperPixelCNNBlock, self).__init__()
        self.mask_type = "B"
        padding = kernel_size // 2
        out_channels = in_channels // 2

        self.main = nn.Sequential(
            HyperConv2d(in_channels, out_channels, 1, activation_fnc="elu", bias=False,
                        bn=True, hyper_config=hyper_config),
            # nn.BatchNorm2d(out_channels),
            # nn.ELU(),
            MaskedConv2d(
                self.mask_type,
                out_channels,
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            HyperConv2d(out_channels, in_channels, 1, bias=False, activation_fnc="none",
                        bn=True, hyper_config=hyper_config),
            # nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        return self.activation(self.main(input) + input)

class HyperPixelCNN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks,
                 kernel_sizes,
                 masked_channels,
                 hyper_config):
        super(HyperPixelCNN, self).__init__()
        assert num_blocks == len(kernel_sizes)
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                block = MaskABlock(in_channels,
                                   out_channels,
                                   kernel_sizes[i],
                                   masked_channels)
            else:
                block = HyperPixelCNNBlock(out_channels, kernel_sizes[i], hyper_config)
            self.blocks.append(block)

        self.main = nn.ModuleList(self.blocks)

        self.direct_connects = []
        for i in range(1, num_blocks - 1):
            self.direct_connects.append(
                HyperPixelCNNBlock(out_channels, kernel_sizes[i], hyper_config))
        self.direct_connects = nn.ModuleList(self.direct_connects)

    def forward(self, input):
        direct_inputs = []
        for i, layer in enumerate(self.main):
            if i > 2:
                direct_input = direct_inputs.pop(0)
                direct_conncet = self.direct_connects[i - 3]
                input = input + direct_conncet(direct_input)

            input = layer(input)
            direct_inputs.append(input)
        assert len(
            direct_inputs) == 3, "architecture error: %d" % len(direct_inputs)
        direct_conncet = self.direct_connects[-1]
        return input + direct_conncet(direct_inputs.pop(0))


class HyperPixelCNNDecoder(BaseHyperDecoder):
    require_inputs = True

    def __init__(self, hyper_config):
        super(HyperPixelCNNDecoder, self).__init__()
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
            HyperPixelCNN(1 + self.fm_latent,
                     hidden_channels,
                     len(kernal_sizes),
                     kernal_sizes,
                     self.nc,
                     hyper_config),
            HyperConv2d(hidden_channels, hidden_channels, 1, activation_fnc="elu", bias=False, bn=True, hyper_config=hyper_config),
            # nn.BatchNorm2d(hidden_channels),
            # nn.ELU(),
            HyperConv2d(hidden_channels, self.nc, 1, bias=False, activation_fnc="none", bn=False, hyper_config=hyper_config),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise ValueError

    # def special_decode(self, z, x):
    #     z = self.z_transform(z)
    #     z = z.view(-1, self.fm_latent, 28, 28)
    #     z = torch.cat([x, z], dim=1)
    #     return self.layers(z)
    #
    # def special_forward(self, z, x):
    #     return self.special_decode(z, x)

    def reconstruct_error(self, x, z):
        eps = 1e-12
        if type(z) == type(None):
            batch_size, nsampels, _, _ = x.size()
            img = x.unsqueeze(1).expand(batch_size, nsampels, *x.size()[1:])
        else:
            z = z.unsqueeze(1)
            batch_size, nsampels, nz = z.size()
            # [batch, nsamples, -1] --> [batch, nsamples, fm, H, W]
            z = self.z_transform(z).view(batch_size, nsampels, self.fm_latent, 28, 28)

            # [batch, nc, H, W] --> [batch, 1, nc, H, W] --> [batch, nsample, nc, H, W]
            img = x.unsqueeze(1).expand(batch_size, nsampels, *x.size()[1:])
            # [batch, nsample, nc+fm, H, W] --> [batch * nsamples, nc+fm, H, W]
            img = torch.cat([img, z], dim=2)

        img = img.view(-1, *img.size()[2:])

        # [batch * nsamples, *] --> [batch, nsamples, -1]
        recon_x = self.forward(img).view(batch_size, nsampels, -1)
        # [batch, -1]
        x_flat = x.view(batch_size, -1)
        BCE = (recon_x + eps).log() * x_flat.unsqueeze(1) + (1.0 - recon_x + eps).log() * (1. - x_flat).unsqueeze(1)
        # [batch, nsamples]
        return BCE.sum(dim=2) * -1.0

