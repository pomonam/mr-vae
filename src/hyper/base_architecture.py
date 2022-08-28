import torch
import torch.nn as nn

from src.hyper.layer import HyperLayer


class BaseHyperEncoder(nn.Module):

  def __init__(self):
    nn.Module.__init__(self)

  def set_net_inputs(self, value: torch.Tensor) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(value)

  def reset_net_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()

  def forward(self, x):
    raise NotImplementedError()


class BaseHyperDecoder(nn.Module):

  def __init__(self):
    nn.Module.__init__(self)

  def set_net_inputs(self, value: torch.Tensor) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(value)

  def reset_net_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()

  def forward(self, z: torch.Tensor):
    raise NotImplementedError()
