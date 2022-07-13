from torch import nn
import torch
from src.models.pixcelcnn import PixelCNN
from torch.autograd import Variable


class MLPDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(64, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 784)

    def forward(self, z):
        z = self.linear1(z)
        z = torch.relu(z)
        z = self.linear2(z)
        z = torch.relu(z)
        z = self.linear3(z)
        z = z.view(z.shape[0], 1, 28, 28)
        return z


class CNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_layer = nn.Linear(64, 32 * 8)

        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(32 * 8, 32 * 4, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32 * 4, 32 * 2, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32 * 2, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        ])

    def forward(self, z):
        z = self.initial_layer(z)
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        for i, layer in enumerate(self.layers):
            z = layer(z)
        return z


class PixelCNNDecoder(nn.Module):
    def __init__(self):
        super(PixelCNNDecoder, self).__init__()
        self.nz = 64
        self.nc = 1
        self.fm_latent = 4

        self.img_latent = 28 * 28 * self.fm_latent
        if self.nz != 0:
            self.z_transform = nn.Sequential(
                nn.Linear(self.nz, self.img_latent),
            )
        kernal_sizes = [7, 7, 7, 5, 5, 3, 3]

        hidden_channels = 32
        self.layers = nn.Sequential(
            PixelCNN(1 + self.fm_latent, hidden_channels, len(kernal_sizes), kernal_sizes, self.nc),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ELU(),
            nn.Conv2d(hidden_channels, self.nc, 1, bias=False),
        )
        # self.reset_parameters()

    def reset_parameters(self):
        if self.nz != 0:
            nn.init.xavier_uniform_(self.z_transform[0].weight)
            nn.init.constant_(self.z_transform[0].bias, 0)

        m = self.layers[2]
        assert isinstance(m, nn.BatchNorm2d)
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    # def _forward(self, z):
    #     # z = z.view(z.shape[0], 64, 1, 1)
    #     z = self.layers(z)
    #     return z

    def forward(self, z):

        H = W = 28
        batch_size, nz = z.size()

        # [batch, -1] --> [batch, fm, H, W]
        z = self.z_transform(z).view(batch_size, self.fm_latent, H, W)
        img = Variable(z.data.new(batch_size, self.nc, H, W).zero_(), volatile=True)
        # [batch, nc+fm, H, W]
        img = torch.cat([img, z], dim=1)
        for i in range(H):
            for j in range(W):
                # [batch, nc, H, W]
                recon_img = self.layers(img)
                # [batch, nc]
                # img[:, :self.nc, i, j] = torch.ge(recon_img[:, :, i, j], 0.5).float() if deterministic else torch.bernoulli(recon_img[:, :, i, j])
                # img[:, :self.nc, i, j] = torch.bernoulli(recon_img[:, :, i, j])
                img[:, :self.nc, i, j] = torch.ge(recon_img[:, :, i, j], 0.5).float()

            # [batch, nc, H, W]
        img_probs = self.layers(img)
        # return img[:, :self.nc], img_probs
        return img_probs

    # def forward(self, z):
    #     z = z.unsqueeze(1)
    #     batch_size, nsampels, nz = z.size()
    #     # [batch, nsamples, -1] --> [batch, nsamples, fm, H, W]
    #     z = self.z_transform(z).view(batch_size, nsampels, self.fm_latent, 28, 28)
    #
    #     # [batch, nc, H, W] --> [batch, 1, nc, H, W] --> [batch, nsample, nc, H, W]
    #     img = x.unsqueeze(1).expand(batch_size, nsampels, *x.size()[1:])
    #     # [batch, nsample, nc+fm, H, W] --> [batch * nsamples, nc+fm, H, W]
    #     img = torch.cat([img, z], dim=2)
    #
    #     img = img.view(-1, *img.size()[2:])
    #     recon_x = self.forward(img).view(batch_size, nsampels, -1)
