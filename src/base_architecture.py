import torch
from torch import nn


class BaseEncoder(nn.Module):

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()


class BaseDecoder(nn.Module):

  def forward(self, inputs: torch.Tensor):
    raise NotImplementedError()
