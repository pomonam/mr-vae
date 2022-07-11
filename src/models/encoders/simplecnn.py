from torch import nn
from src.utils import load_activation
from typing import Tuple
from src.models.encoders.base import BaseEncoder
from src.utils import get_conv_layers

class SimpleCNNEncoder(BaseEncoder):
    def __init__(self,
                 activation: str = "relu",
                 bias: bool = True):
        super().__init__()

        self.layers = nn.ModuleList(get_conv_layers(nn.Conv2d, [1, 32, 64, 128], [4, 4, 4, 4], [2, 2, 2, 1], [1, 1, 1, 0], [False, False, False, False], [True, True, True, True]))
        # self.layers = nn.ModuleList([
        #     nn.Conv2d(1, 32, kernel_size=5, stride=1, padding="same"),
        #     nn.Conv2d(32, 32, kernel_size=5, stride=2, padding="same"),
        #     nn.Conv2d(32, 64, kernel_size=5, stride=1, padding="same"),
        #     nn.Conv2d(64, 64, kernel_size=5, stride=2, padding="same"),
        #     nn.Conv2d(64, 256, kernel_size=7, stride=1, padding="valid"),
        # ])
        self.activation_fn = load_activation(name=activation)

    def forward(self, x, *argv):
        for i, layer in enumerate(self.layers):
            x = layer(x, *argv)
            x = self.activation_fn(x)
        return x
