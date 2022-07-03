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


class MLPDecoder(nn.Module):
  def __init__(self, latent_size, structure=(784, 70, 10), activation="relu", bias=True):
    super().__init__()

    self.layers = nn.ModuleList([
      nn.Linear(a, b, bias=bias)
      for a, b in zip(structure, structure[1:])
    ])
    self.activation_fn = load_activation(name=activation)

    self.mean = nn.Linear(structure[-1], latent_size, bias=bias)
    self.log_stddev = nn.Linear(structure[-1], latent_size, bias=bias)

  def forward(self, z):
    z = z.view(-1, self.structure[0])
    for i, layer in enumerate(self.layers):
      z = layer(z)
      if i != len(self.layers) - 1:
        z = self.activation_fn(z)
    return z

