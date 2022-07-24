import copy

import torch
from torch import nn

from src.config import HyperConfig
from src.hyper_models.blocks import get_block
from src.hyper_models.module import HyperModule


class HyperLinear(HyperModule):
    def __init__(self,
                 module: nn.Module,
                 hyper_config: HyperConfig):
        super().__init__(module, hyper_config)

        assert isinstance(module, nn.Linear)
        self.module = copy.deepcopy(module)
        self.width = module.out_features
        self.hyper_module = get_block(self.block_type)(self.output_dim + 1, hyper_config)
        self.bias = nn.Parameter(torch.zeros(self.output_dim, 1))

        if self.hyper_type == "add":
            self.add_module = copy.deepcopy(module)
            self.add_bias = nn.Parameter(torch.zeros(self.output_dim, 1))
        else:
            self.add_module = None

        self.bias = nn.Parameter(torch.zeros(self.output_dim, 1))
        self.output_layer = nn.Linear(self.output_dim, self.output_dim, bias=False)

    def forward(self, inputs):
        hyper_out = self.hyper_module(self._beta)
        hyper_weight = hyper_out[:, :-1]
        hyper_bias = hyper_out[:, -1]
        if self.hyper_type == "add":
            out = self.module(inputs)
            temp = self.add_module(inputs) * hyper_weight
            if self.include_output_layer:
                out = out + self.output_layer(temp)
            else:
                out = out + temp
            out = out + self.bias 

        elif self.hyper_type == "s_add":
            out = self.module(inputs)
            temp = self.module(inputs) * hyper_weight
            if self.include_output_layer:
                out = out + self.output_layer(temp)
            else:
                out = out + temp

        elif self.hyper_type == "mult":
            out = self.module(inputs) * hyper_weight
            if self.include_output_layer:
                out = self.output_layer(out)
