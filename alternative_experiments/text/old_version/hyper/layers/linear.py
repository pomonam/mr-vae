import torch
import torch.nn as nn

from src.config import HyperConfig
from alternative_experiments.text.old_version.hyper.layers.blocks import get_block
from alternative_experiments.text.old_version.hyper.layers.module import get_activation
from alternative_experiments.text.old_version.hyper.layers.module import HyperModule


class HyperLinear(HyperModule):

    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features, bias=True)
        self.hyper_block_scale = get_block("linear")(1, self.out_features)

    def forward(self, inputs):
        scale = self.hyper_block_scale(self._net_inputs)
        scale = torch.sigmoid(scale)
        return scale * self.linear(inputs)
