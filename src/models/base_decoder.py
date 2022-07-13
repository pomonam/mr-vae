import torch
import torch.nn as nn


class BaseDecoder(nn.Module):

  def decode(self,
             x: torch.Tensor,
             z: torch.Tensor,
             *argv) -> torch.Tensor:
    raise NotImplementedError

  def reconstruct_error(self,
                        x: torch.Tensor,
                        z: torch.Tensor,
                        *argv) -> torch.Tensor:
    raise NotImplementedError

  def beam_search_decode(self, z: torch.Tensor, k: int):
    raise NotImplementedError

  def sample_decode(self, z: torch.Tensor):
    raise NotImplementedError

  def greedy_decode(self, z: torch.Tensor) -> list:
    raise NotImplementedError

  def log_probability(self,
                      x: torch.Tensor,
                      z: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError
