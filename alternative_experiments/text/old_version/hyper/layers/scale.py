import torch

from src.config import HyperConfig
from alternative_experiments.text.old_version.hyper.layers.blocks import get_block
from alternative_experiments.text.old_version.hyper.layers.module import get_activation
from alternative_experiments.text.old_version.hyper.layers.module import HyperModule


class HyperScale(HyperModule):

    def __init__(self,
                 out_features: int):
        super().__init__()

        self.out_features = out_features

        input_dim = 1
        self.hyper_block_scale = get_block("linear")(input_dim, self.out_features)

    def forward(self, inputs):
        scale = self.hyper_block_scale(self._net_inputs)
        scale = torch.sigmoid(scale)
        return inputs * scale.unsqueeze(1)
