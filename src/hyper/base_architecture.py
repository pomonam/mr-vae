import torch

from src.base_architecture import BaseDecoder
from src.base_architecture import BaseEncoder
from src.hyper.layers import HyperLayer


class BaseHyperEncoder(BaseEncoder):

  def set_inputs(self,
                 net_inputs: torch.Tensor,
                 beta_inputs: torch.Tensor) -> None:
    is_triggered = False
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(net_inputs)
        module.set_beta_inputs(beta_inputs)
        is_triggered = True
    if not is_triggered:
      print("Warning: No hyper_params registered for encoder.")

  def reset_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()
        module.reset_beta_inputs()

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()


class BaseHyperDecoder(BaseDecoder):

  def set_inputs(self,
                 net_inputs: torch.Tensor,
                 beta_inputs: torch.Tensor) -> None:
    is_triggered = False
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.set_net_inputs(net_inputs)
        module.set_beta_inputs(beta_inputs)
        is_triggered = True
    if not is_triggered:
      print("Warning: No hyper_params registered for encoder.")

  def reset_inputs(self) -> None:
    for module in self.modules():
      if isinstance(module, HyperLayer):
        module.reset_net_inputs()
        module.reset_beta_inputs()

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()
