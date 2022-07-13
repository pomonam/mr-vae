import math
import torch
import torch.nn as nn

from src.utils import log_sum_exp


class BaseEncoder(nn.Module):
  """docstring for EncoderBase"""

  def forward(self, x):
    raise NotImplementedError
