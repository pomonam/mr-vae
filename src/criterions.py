import torch
import torch.nn.functional as F


def binary_cross_entropy(x, logits):
    """Calculate binary (logistic) cross-entropy from distribution logits.
    Args:
      x: input variable tensor, must be of same shape as logits
      logits: log odds of a Bernoulli distribution, i.e. log(p/(1-p))
    Returns:
      A scalar representing binary CE for the given Bernoulli distribution.
    """
    if x.shape != logits.shape:
        raise ValueError("inputs x and logits must be of the same shape")

    x = torch.reshape(x, (x.shape[0], -1))
    logits = torch.reshape(logits, (logits.shape[0], -1))

    bce_loss = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
    return torch.sum(bce_loss, dim=-1)


def kl_gaussian(mean, var):
    r"""Calculate KL divergence between given and standard gaussian distributions.
    KL(p, q) = H(p, q) - H(p) = -\int p(x)log(q(x))dx - -\int p(x)log(p(x))dx
             = 0.5 * [log(|s2|/|s1|) - 1 + tr(s1/s2) + (m1-m2)^2/s2]
             = 0.5 * [-log(|s1|) - 1 + tr(s1) + m1^2] (if m2 = 0, s2 = 1)
    Args:
      mean: mean vector of the first distribution
      var: diagonal vector of covariance matrix of the first distribution
    Returns:
      A scalar representing KL divergence of the two Gaussian distributions.
    """
    return 0.5 * torch.sum(-torch.log(var) - 1.0 + var + torch.square(mean), dim=-1)

# class BetaElbo(nn.Module):
#     def __init__(self):
#         pass

