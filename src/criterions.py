import torch
import torch.nn.functional as F


def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(
            torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def calc_au(model, test_loader, delta=0.01):
    means = []
    for datum in test_loader:
        output_dict = model(datum["inputs"])
        means.append(output_dict["mean"])

    means = torch.cat(means, dim=0)
    au_mean = means.mean(0, keepdim=True)

    # (batch_size, nz)
    au_var = means - au_mean
    ns = au_var.size(0)

    au_var = (au_var**2).sum(dim=0) / (ns - 1)

    return (au_var >= delta).sum().item(), au_var


def binary_cross_entropy(x, logits):
    if x.shape != logits.shape:
        raise ValueError("Inputs x and logits must be of the same shape")

    x = torch.reshape(x, (x.shape[0], -1))
    logits = torch.reshape(logits, (logits.shape[0], -1))

    bce_loss = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return torch.sum(bce_loss, dim=-1)


def kl_gaussian(mean, var):
    return 0.5 * torch.sum(
        -torch.log(var) - 1.0 + var + torch.square(mean), dim=-1)
