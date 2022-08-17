import math
import torch.nn as nn

import torch
from torch.nn import init
import torch.nn.functional as F
from src.hyper.layers.module import get_activation

from src.config import HyperConfig
from src.hyper.layers.blocks import get_block
from src.hyper.layers.module import HyperModule


class HyperConv2d(HyperModule):

    def __init__(self, in_channels, out_channels, kernel_size, activation_fnc, hyper_config: HyperConfig, stride=1,
                 padding=0, dialation=1, group=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dialation = dialation
        self.group = group
        self.bias = bias
        self.activation_fnc = get_activation(activation_fnc)
        self.hyper_config = hyper_config

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if self.hyper_config.include_chunk:
            self.chunk_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
        self.hyper_block_scale = get_block("linear")(input_dim, self.out_channels)
        self.hyper_block_shift = get_block("linear")(input_dim, self.out_channels)
        if self.hyper_config.include_chunk:
            self.chunk_hyper_block_scale = get_block("linear")(input_dim, self.out_channels)
            self.chunk_hyper_block_shift = get_block("linear")(input_dim, self.out_channels)
            self.chunk_moe = get_block("linear")(input_dim, 2)

        if self.hyper_config.include_layer_norm:
            self.layer_norm = torch.nn.GroupNorm(1, self.out_channels, affine=False)

    def forward(self, inputs):
        scale = self.hyper_block_scale(self._net_inputs).unsqueeze(-1).unsqueeze(-1)
        if self.hyper_config.include_sigmoid_activation:
            scale = torch.sigmoid(scale)
        shift = self.hyper_block_shift(self._net_inputs).unsqueeze(-1).unsqueeze(-1)

        if self.hyper_config.include_chunk:
            chunk_scale = self.chunk_hyper_block_scale(self._net_inputs).unsqueeze(-1).unsqueeze(-1)
            if self.hyper_config.include_sigmoid_activation:
                chunk_scale = torch.sigmoid(chunk_scale)
            chunk_shift = self.chunk_hyper_block_shift(self._net_inputs).unsqueeze(-1).unsqueeze(-1)

        pre_act = self.conv2d(inputs)
        if self.hyper_config.include_chunk:
            chunk_pre_act = self.chunk_conv2d(inputs)

        if self.hyper_config.preact_transform:
            if self.hyper_config.include_layer_norm:
                hyper_pre_act = scale * self.layer_norm(pre_act) + shift
            else:
                hyper_pre_act = scale * pre_act + shift

            if self.hyper_config.include_chunk:
                if self.hyper_config.include_layer_norm:
                    chunk_hyper_pre_act = chunk_scale * self.layer_norm(chunk_pre_act) + chunk_shift
                else:
                    chunk_hyper_pre_act = chunk_scale * chunk_pre_act + chunk_shift

                chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
                hyper_pre_act = hyper_pre_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) +\
                                    chunk_hyper_pre_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                pre_act = pre_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
                            chunk_pre_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

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
                    chunk_hyper_act = chunk_scale * self.layer_norm(chunk_act) + chunk_shift
                else:
                    chunk_hyper_act = chunk_scale * chunk_act + chunk_shift

                chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
                hyper_act = hyper_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
                                chunk_hyper_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                act = act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
                        chunk_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if self.hyper_config.include_residual_connection:
                return act + hyper_act
            else:
                return hyper_act

        # else:
        # # out = inputs + inputs * hyper_weight + hyper_bias
        # out = inputs + inputs * hyper_weight.unsqueeze(-1).unsqueeze(
        #     -1) + hyper_bias.unsqueeze(-1).unsqueeze(-1)
        #
        # return out


class HyperConvTranspose2d(HyperModule):

    def __init__(self, in_channels, out_channels, kernel_size, activation_fnc, hyper_config: HyperConfig, stride=1,
                 padding=0, dialation=1, group=1, output_padding=0, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dialation = dialation
        self.group = group
        self.bias = bias
        self.activation_fnc = get_activation(activation_fnc)
        self.hyper_config = hyper_config

        self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, output_padding=output_padding)
        if self.hyper_config.include_chunk:
            self.chunk_conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, output_padding=output_padding)

        input_dim = hyper_config.preprocess_dim if hyper_config.preprocess_beta else 1
        self.hyper_block_scale = get_block("linear")(input_dim, self.out_channels)
        self.hyper_block_shift = get_block("linear")(input_dim, self.out_channels)
        if self.hyper_config.include_chunk:
            self.chunk_hyper_block_scale = get_block("linear")(input_dim, self.out_channels)
            self.chunk_hyper_block_shift = get_block("linear")(input_dim, self.out_channels)
            self.chunk_moe = get_block("linear")(input_dim, 2)

        if self.hyper_config.include_layer_norm:
            self.layer_norm = torch.nn.GroupNorm(1, self.out_channels, affine=False)

    def forward(self, inputs):
        scale = self.hyper_block_scale(self._net_inputs).unsqueeze(-1).unsqueeze(-1)
        if self.hyper_config.include_sigmoid_activation:
            scale = torch.sigmoid(scale)
        shift = self.hyper_block_shift(self._net_inputs).unsqueeze(-1).unsqueeze(-1)

        if self.hyper_config.include_chunk:
            chunk_scale = self.chunk_hyper_block_scale(self._net_inputs).unsqueeze(-1).unsqueeze(-1)
            if self.hyper_config.include_sigmoid_activation:
                chunk_scale = torch.sigmoid(chunk_scale)
            chunk_shift = self.chunk_hyper_block_shift(self._net_inputs).unsqueeze(-1).unsqueeze(-1)

        pre_act = self.conv2d(inputs)
        if self.hyper_config.include_chunk:
            chunk_pre_act = self.chunk_conv2d(inputs)

        if self.hyper_config.preact_transform:
            if self.hyper_config.include_layer_norm:
                hyper_pre_act = scale * self.layer_norm(pre_act) + shift
            else:
                hyper_pre_act = scale * pre_act + shift

            if self.hyper_config.include_chunk:
                if self.hyper_config.include_layer_norm:
                    chunk_hyper_pre_act = chunk_scale * self.layer_norm(chunk_pre_act) + chunk_shift
                else:
                    chunk_hyper_pre_act = chunk_scale * chunk_pre_act + chunk_shift

                chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
                hyper_pre_act = hyper_pre_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) +\
                                    chunk_hyper_pre_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                pre_act = pre_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)+\
                            chunk_pre_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

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
                    chunk_hyper_act = chunk_scale * self.layer_norm(chunk_act) + chunk_shift
                else:
                    chunk_hyper_act = chunk_scale * chunk_act + chunk_shift

                chunk_moe = torch.softmax(self.chunk_moe(self._net_inputs), 1)
                hyper_act = hyper_act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
                                chunk_hyper_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                act = act * chunk_moe[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + \
                        chunk_act * chunk_moe[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            if self.hyper_config.include_residual_connection:
                return act + hyper_act
            else:
                return hyper_act
