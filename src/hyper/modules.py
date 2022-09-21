import torch
from torch import nn

from src.hyper.layers import HyperLayer


class HyperLinear(HyperLayer):

  def __init__(self,
               in_features: int,
               out_features: int,
               bias=True,
               decoder=False) -> None:
    super().__init__()

    self.layer = nn.Linear(in_features, out_features, bias)
    self.decoder = decoder
    self.hyper_layer = nn.Linear(1, out_features, bias=True)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    out = self.layer(inputs)
    scale = self.hyper_layer(self._net_inputs)

    if self.decoder:
      scale = torch.relu(1 - torch.exp(scale))
      scale = torch.sqrt(scale)
    else:
      scale = torch.sigmoid(scale)
    return scale * out


class HyperConv2d(HyperLayer):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               decoder=False):
    super().__init__()

    self.layer = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           dilation,
                           groups,
                           bias)
    self.decoder = decoder
    self.hyper_layer = nn.Linear(1, out_channels, bias=True)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    out = self.layer(inputs)
    scale = self.hyper_layer(self._net_inputs)

    if self.decoder:
      scale = torch.relu(1 - torch.exp(scale))
      scale = torch.sqrt(scale)
    else:
      scale = torch.sigmoid(scale)
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    return scale * out


class HyperConvTranspose2d(HyperLayer):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               output_padding=0,
               groups=1,
               bias=True,
               decoder=False):
    super().__init__()

    self.layer = nn.ConvTranspose2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    output_padding,
                                    groups,
                                    bias)
    self.decoder = decoder
    self.hyper_layer = nn.Linear(1, out_channels, bias=True)

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    out = self.layer(inputs)
    scale = self.hyper_layer(self._net_inputs)

    if self.decoder:
      scale = torch.relu(1 - torch.exp(scale))
      scale = torch.sqrt(scale)
    else:
      scale = torch.sigmoid(scale)
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    return scale * out
