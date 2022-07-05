from torch import nn

from .utils import load_activation


class MLPEncoder(nn.Module):
    def __init__(self, structure=(784, 70, 10), activation="relu", bias=True):
        super().__init__()

        self.structure = structure
        self.layers = nn.ModuleList([
            nn.Linear(a, b, bias=bias)
            for a, b in zip(structure, structure[1:])
        ])
        self.activation_fn = load_activation(name=activation)

    def forward(self, z, *argv):
        z = z.view(-1, self.structure[0])
        for i, layer in enumerate(self.layers):
            z = layer(z, *argv)
            if i != len(self.layers) - 1:
                z = self.activation_fn(z)
        return z
