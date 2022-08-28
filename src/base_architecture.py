import torch
import torch.nn as nn


class BaseEncoder(nn.Module):

  def __init__(self):
    nn.Module.__init__(self)

  def forward(self, x):
    raise NotImplementedError()


class BaseDecoder(nn.Module):

  def __init__(self):
    nn.Module.__init__(self)

  def forward(self, z: torch.Tensor):
    raise NotImplementedError()
