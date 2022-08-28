import functools

import torch
from torch import nn
import torch.nn.functional as F


def get_activation(act_name):

  def identity(x):
    return x

  act_dict = {
      "relu": F.relu,
      "none": identity,
      "elu": F.elu,
      "leaky_relu": functools.partial(F.leaky_relu, negative_slope=0.2),
  }
  return act_dict[act_name]


class HyperModule(nn.Module):

  def __init__(self):
    super().__init__()
    self._net_inputs = None

  def set_net_inputs(self, value: torch.Tensor) -> None:
    self._net_inputs = value

  def reset_net_inputs(self) -> None:
    self._net_inputs = None
