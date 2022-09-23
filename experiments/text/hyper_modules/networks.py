# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Various neural networks and related utilities.
"""
from torch import nn

from texar.torch.modules.networks.network_base import FeedForwardNetworkBase
from texar.torch.utils.utils import get_output_size
from texar.torch.core.layers import get_layer
from texar.torch.hyperparams import HParams
from texar.torch.module_base import ModuleBase
from texar.torch.utils.utils import uniquify_str
from src.hyper.layers import get_hyper_layer


class HyperFeedForwardNetwork(FeedForwardNetworkBase):
    def __init__(self, layers=None, hparams=None, hyper_cfg=None):
        super().__init__(hparams=hparams)
        self.hyper_cfg = hyper_cfg
        self._build_layers(layers=layers,
                           layer_hparams=self._hparams.layers,
                           hyper_cfg=self.hyper_cfg)
        self.hyper1 = get_hyper_layer(1024, hyper_cfg, decoder=True)
        self.hyper2 = get_hyper_layer(256, hyper_cfg, decoder=True)

    @staticmethod
    def default_hparams():
        return {
            "layers": [],
            "name": "NN"
        }

    @property
    def output_size(self) -> int:
        r"""The feature size of network layers output. If output size is
        only determined by input, the feature size is equal to ``-1``.
        """
        for i, layer in enumerate(reversed(self._layers)):
            size = get_output_size(layer)
            size_ext = getattr(layer, 'output_size', None)
            if size_ext is not None:
                size = size_ext
            if size is None:
                break
            if size > 0:
                return size
            elif i == len(self._layers) - 1:
                return -1

        raise ValueError("'output_size' can not be calculated because "
                         "'FeedForwardNetwork' contains submodule "
                         "whose output size cannot be determined.")


    def _build_layers(self,
                      layers,
                      layer_hparams,
                      hyper_cfg):
      if layers is not None:
        self._layers = layers
      else:
        if layer_hparams is None:
          raise ValueError(
            'Either `layer` or `layer_hparams` is required.')
        self._layers = nn.ModuleList()
        for _, hparams in enumerate(layer_hparams):
          self._layers.append(get_layer(hparams=hparams))

      for layer in self._layers:
        layer_name = uniquify_str(layer.__class__.__name__,
                                  self._layer_names)
        self._layer_names.append(layer_name)
        self._layers_by_name[layer_name] = layer

    def forward(self,  # type: ignore
                input):
        r"""Feeds forward inputs through the network layers and returns outputs.

        Args:
            input: The inputs to the network. The requirements on inputs
                depends on the first layer and subsequent layers in the
                network.

        Returns:
            The output of the network.
        """
        outputs = input
        i = 0
        for layer in self._layers:
            outputs = layer(outputs)
            if i == 1:
              outputs = self.hyper1(outputs)
            i += 1
        outputs = self.hyper2(outputs)
        return outputs
