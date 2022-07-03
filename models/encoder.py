import torch
import torch.nn as nn
import torch.nn.functional as F


def load_activation(name):
    activations = {
        "id": lambda z: z,
        "relu": F.relu,
        "tanh": torch.tanh,
    }
    return activations[name]


class MLPEncoder(nn.Module):
  def __init__(self, latent_size, structure=(784, 70, 10), activation="relu", bias=True):
    super().__init__()

    self.layers = nn.ModuleList([
      nn.Linear(a, b, bias=bias)
      for a, b in zip(structure, structure[1:])
    ])
    self.activation_fn = load_activation(name=activation)

    self.mean = nn.Linear(structure[-1], latent_size, bias=bias)
    self.log_stddev = nn.Linear(structure[-1], latent_size, bias=bias)

  def forward(self, x):
    x = x.view(-1, self.structure[0])
    for i, layer in enumerate(self.layers):
      x = layer(x)
      x = self.activation_fn(x)

    mean = self.mean(x)
    log_stddev = self.log_stddev(x)
    stddev = torch.exp(log_stddev)
    return mean, stddev

