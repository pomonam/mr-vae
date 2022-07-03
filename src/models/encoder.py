import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_activation


class MLPEncoder(nn.Module):
    def __init__(self, structure=(784, 70, 10), activation="relu", bias=True):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Linear(a, b, bias=bias)
            for a, b in zip(structure, structure[1:])
        ])
        self.activation_fn = load_activation(name=activation)

    def forward(self, x):
        x = x.view(-1, self.structure[0])
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.activation_fn(x)
        return x
        # mean = self.mean(x)
        # log_stddev = self.log_stddev(x)
        # stddev = torch.exp(log_stddev)
        # return mean, stddev
