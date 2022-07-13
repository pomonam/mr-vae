from torch import nn
import torch
import math
from src.models.resnet import ResNet


class MLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)

    def forward(self, x):
        x = x.view(x.shape[0], 784)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32 * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32 * 2, 32 * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32 * 4, 32 * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            ResNet(1, [64, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Flatten()
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        return out
