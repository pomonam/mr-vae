import torch
import torch.nn.functional as F


def load_activation(name):
    activations = {
        "id": lambda z: z,
        "relu": F.relu,
        "tanh": torch.tanh,
    }
    return activations[name]
