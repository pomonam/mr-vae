import torch
from torch import nn

from src.base_architecture import BaseEncoder


class MlpEncoder(BaseEncoder):

  def __init__(self):
    super().__init__()

    self.input_dim = (1, 28, 28)
    self.latent_dim = 32

    layers = nn.ModuleList()
    layers.append(
      nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU()
      )
    )

    self.layers = layers
    self.depth = len(layers)

    self.embedding = nn.Linear(1024, self.latent_dim)
    self.log_var = nn.Linear(1024, self.latent_dim)

  def forward(self, inputs: torch.Tensor) -> dict:
    max_depth = self.depth
    out = inputs.reshape(-1, 784)
    output = {}
    for i in range(max_depth):
      out = self.layers[i](out)

      if i + 1 == self.depth:
        output["embedding"] = self.embedding(out)
        output["log_covariance"] = self.log_var(out)

    return output
