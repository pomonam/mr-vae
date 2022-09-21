import torch
import torch.nn as nn

from src.config import HyperConfig
from experiments.text_legacy.old_version.hyper.layers.blocks import get_block
from experiments.text_legacy.old_version.hyper.layers.module import get_activation
from experiments.text_legacy.old_version.hyper.layers.module import HyperModule


class HyperScale(HyperModule):

    def __init__(self,
                 out_features: int,
                 activation_fnc: str,
                 hyper_config: HyperConfig):
        super().__init__()

        self.out_features = out_features
        self.activation_fnc = get_activation(activation_fnc)
        self.hyper_config = hyper_config

        input_dim = 1
        self.hyper_block_scale = get_block("linear")(input_dim, self.out_features)
        self.hyper_block_shift = get_block("linear")(input_dim, self.out_features)


    def forward(self, inputs):
        scale = self.hyper_block_scale(self._net_inputs)
        scale = torch.sigmoid(scale)
        return inputs * scale.unsqueeze(1)
