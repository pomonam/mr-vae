import torch
import torch.nn as nn

from src.config import HyperConfig
from experiments.text_legacy.old_version.hyper.layers.blocks import get_block
from experiments.text_legacy.old_version.hyper.layers.module import get_activation
from experiments.text_legacy.old_version.hyper.layers.module import HyperModule


class HyperLinear(HyperModule):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation_fnc: str,
                 hyper_config: HyperConfig,
                 bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fnc = get_activation(activation_fnc)
        self.hyper_config = hyper_config
        self.bias = bias

        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        input_dim = 1
        self.hyper_block_scale = get_block("linear")(input_dim, self.out_features)
        self.hyper_block_shift = get_block("linear")(input_dim, self.out_features)

    def forward(self, inputs):
        scale = self.hyper_block_scale(self._net_inputs)
        scale = torch.sigmoid(scale)
        return self.activation_fnc(scale * self.linear(inputs))
