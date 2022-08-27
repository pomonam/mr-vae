import torch
import torch.nn as nn

from src.config import HyperConfig
from src.hyper.layers.blocks import get_block
from src.hyper.layers.module import get_activation
from src.hyper.layers.module import HyperModule


class HyperScale(HyperModule):

    def __init__(self,
                 out_features: int,
                 activation_fnc: str,
                 hyper_config: HyperConfig):
        super().__init__()

        self.out_features = out_features
        self.activation_fnc = get_activation(activation_fnc)
        self.hyper_config = hyper_config

        input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
        self.hyper_block_scale = get_block("linear")(input_dim,
                                                     self.out_features)
        self.hyper_block_shift = get_block("linear")(input_dim,
                                                     self.out_features)
        if self.hyper_config.include_chunk:
            self.chunk_hyper_block_scale = get_block("linear")(
                input_dim, self.out_features)
            self.chunk_hyper_block_shift = get_block("linear")(
                input_dim, self.out_features)
            self.chunk_moe = get_block("linear")(input_dim, 2)

        if self.hyper_config.include_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(
                self.out_features, elementwise_affine=False)

    def forward(self, inputs):
        scale = self.hyper_block_scale(self._net_inputs)
        if self.hyper_config.include_sigmoid_activation:
            scale = torch.sigmoid(scale)
        shift = self.hyper_block_shift(self._net_inputs)

        if self.hyper_config.include_chunk:
            chunk_scale = self.chunk_hyper_block_scale(self._net_inputs)
            if self.hyper_config.include_sigmoid_activation:
                chunk_scale = torch.sigmoid(chunk_scale)
            chunk_shift = self.chunk_hyper_block_shift(self._net_inputs)

        pre_act = inputs
        chunk_pre_act = inputs
        if self.hyper_config.preact_transform:
            if self.hyper_config.include_layer_norm:
                hyper_pre_act = scale * self.layer_norm(pre_act) + shift
            else:
                hyper_pre_act = scale * pre_act + shift

            if self.hyper_config.include_chunk:
                if self.hyper_config.include_layer_norm:
                    chunk_hyper_pre_act = chunk_scale * self.layer_norm(
                        chunk_pre_act) + chunk_shift
                else:
                    chunk_hyper_pre_act = chunk_scale * chunk_pre_act + chunk_shift

                chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
                hyper_pre_act = hyper_pre_act * chunk_moe[:, 0].unsqueeze(-1) +\
                                    chunk_hyper_pre_act * chunk_moe[:, 1].unsqueeze(-1)
                pre_act = pre_act * chunk_moe[:, 0].unsqueeze(-1) +\
                            chunk_pre_act * chunk_moe[:, 1].unsqueeze(-1)

            if self.hyper_config.include_residual_connection:
                return self.activation_fnc(pre_act + hyper_pre_act)
            else:
                return self.activation_fnc(hyper_pre_act)

        else:
            act = self.activation_fnc(pre_act)
            if self.hyper_config.include_layer_norm:
                hyper_act = scale * self.layer_norm(act) + shift
            else:
                hyper_act = scale * act + shift

            if self.hyper_config.include_chunk:
                chunk_act = self.activation_fnc(chunk_pre_act)
                if self.hyper_config.include_layer_norm:
                    chunk_hyper_act = chunk_scale * self.layer_norm(
                        chunk_act) + chunk_shift
                else:
                    chunk_hyper_act = chunk_scale * chunk_act + chunk_shift

                chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
                hyper_act = hyper_act * chunk_moe[:, 0].unsqueeze(-1) + \
                                chunk_hyper_act * chunk_moe[:, 1].unsqueeze(-1)
                act = act * chunk_moe[:, 0].unsqueeze(-1) + \
                        chunk_act * chunk_moe[:, 1].unsqueeze(-1)
            if self.hyper_config.include_residual_connection:
                return act + hyper_act
            else:
                return hyper_act

        # if self.hyper_config.include_sigmoid_activation == "none":
        #   scale = scale
        # elif self.hyper_config.hyper_activation == "sigmoid":
        #   scale = torch.sigmoid(scale)
        # elif self.hyper_config.hyper_activation == "tanh":
        #   scale = torch.tanh(scale)
        # else:
        #   raise NotImplementedError
        #
        # if self.hyper_config.include_layer_norm:
        #   act = self.layer_norm(inputs)
        # else:
        #   act = inputs
        #
        # act = scale * act + shift
        #
        # if self.hyper_config.include_residual_connection:
        #   return inputs + act
        # else:
        #   return act
