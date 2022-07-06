import torch
import torch.nn.functional as F


def binary_cross_entropy(x, logits):
    if x.shape != logits.shape:
        raise ValueError("inputs x and logits must be of the same shape")

    x = torch.reshape(x, (x.shape[0], -1))
    logits = torch.reshape(logits, (logits.shape[0], -1))

    bce_loss = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return torch.sum(bce_loss, dim=-1)


def kl_gaussian(mean, var):
    return 0.5 * torch.sum(-torch.log(var) - 1.0 + var + torch.square(mean), dim=-1)

