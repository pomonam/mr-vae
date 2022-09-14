import torch

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.hyper.layers import HyperLayer


class BaseHyperEncoder(BaseEncoder):

  def set_net_inputs(self, value: torch.Tensor) -> None:
    is_triggered = False
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(value)
        is_triggered = True
    if not is_triggered:
      print("Warning: No hyper_params registered for encoder.")

  def reset_net_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()


class BaseHyperDecoder(BaseDecoder):

  def set_net_inputs(self, value: torch.Tensor) -> None:
    is_triggered = False
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(value)
        is_triggered = True
    if not is_triggered:
      print("Warning: No hyper_params registered for decoder.")

  def reset_net_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()
