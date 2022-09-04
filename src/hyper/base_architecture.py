import torch
from torch import nn
from src.hyper.layers import HyperLayer


class BaseHyperEncoder(nn.Module):

  def __init__(self) -> None:
    super().__init__()

  def set_net_inputs(self, value: torch.Tensor) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(value)

  def reset_net_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()


class BaseHyperDecoder(nn.Module):

  def __init__(self) -> None:
    super().__init__()

  def set_net_inputs(self, value: torch.Tensor) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(value)

  def reset_net_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()
