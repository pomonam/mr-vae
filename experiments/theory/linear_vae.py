import math
from math import pi

import torch
import torch.nn as nn

from src.config import HyperConfig
from src.hyper.layers.linear import HyperLinear
from src.hyper.layers.module import HyperModule
from src.hyper.transformations import stretch_sigmoid
from src.hyper.transformations import stretch_sigmoid_inv

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

# ================================================================================================
# Linear VAE Class Adapted From
# GitHub: https://github.com/BorealisAI/rate_distortion/blob/master/rate_distortion/models/vaes.py
# ================================================================================================


def log_normal_likelihood(x, mean, logvar):
  """Implementation WITHOUT constant
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    Args:
        x: [B,Z]
        mean,logvar: [B,Z]
    Returns:
        output: [B]
    """

  dim = list(mean.size())[1]
  logvar = (torch.zeros(mean.size()) + logvar).to(DEVICE)
  return -0.5 * ((logvar + (x - mean)**2 / torch.exp(logvar)).sum(1))


def log_mean_exp(x, dim=1):
  """ based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    """
  max_, _ = torch.max(x, dim, keepdim=True, out=None)
  return torch.log(torch.mean(torch.exp(x - max_), dim)) + torch.squeeze(max_)


def log_normal(x, mean, logvar):
  """
    based on https://github.com/lxuechen/BDMC/blob/master/utils.py
    log normal WITHOUT constant, since the constants in p(z)
    and q(z|x) cancels out later
    Args:s
        x: [B,Z]
        mean,logvar: [B,Z]
    Returns:
        output: [B]
    """
  return -0.5 * (logvar.sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))


def singleton_repeat(x, n):
  """ 
    based on https://github.com/BorealisAI/rate_distortion/blob/master/rate_distortion/utils/computation_utils.py
    Repeat a batch of data n times. 
    It's the safe way to repeat
    First add an additional dimension, repeat that dimention, then reshape it back. 
    So that later when reshaping, it's guranteed to follow the same tensor convention. 
     """
  if n == 1:
    return x
  else:
    singleton_x = torch.unsqueeze(x, 0)
    repeated_x = singleton_x.repeat(n, 1, 1)
    return repeated_x.view(-1, x.size()[-1])


class LinearEncoder(nn.Module):

  def __init__(self, bottleneck_size):
    super().__init__()
    self.bottleneck_size = bottleneck_size
    self.layer = nn.Linear(784, bottleneck_size * 2, bias=False)

  def forward(self, x):
    x = x.reshape(-1, 784)
    return self.layer(x)


class LinearDecoder(nn.Module):

  def __init__(self, bottleneck_size):
    super().__init__()
    self.bottleneck_size = bottleneck_size
    self.layer = nn.Linear(bottleneck_size, 784, bias=False)

  def forward(self, x):
    x = x.reshape(-1, self.bottleneck_size)
    return self.layer(x)


class LinearVae(nn.Module):

  def __init__(self, encoder, decoder):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.observation_log_likelihood_fn = log_normal_likelihood
    self.x_logvar = nn.Parameter(torch.log(torch.tensor(1)), requires_grad=True)

  def encode(self, x):
    hidden = self.encoder(x)
    mean = hidden[:, :self.encoder.bottleneck_size]
    logvar = hidden[:, self.encoder.bottleneck_size:]
    return mean, logvar

  def decode(self, z):
    return self.decoder(z), torch.zeros(1)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = eps * std + mu
    logqz = log_normal(z, mu, logvar)
    zeros = torch.zeros_like(z)
    logpz = log_normal(z, zeros, zeros)
    return z, logpz, logqz

  def forward(self, x, num_iwae=1):
    flattened_x = x.view(-1, 784)
    flattened_x_k = singleton_repeat(flattened_x, num_iwae)
    mu, logvar = self.encode(flattened_x_k)
    z, logpz, logqz = self.reparameterize(mu, logvar)
    x_mean, x_logvar = self.decode(z)

    likelihood = self.observation_log_likelihood_fn(flattened_x_k,
                                                    x_mean,
                                                    x_logvar)
    elbo = likelihood + logpz - logqz

    if num_iwae != 1:
      elbo = log_mean_exp(elbo.view(num_iwae, -1), dim=0)
      logpz = log_mean_exp(logpz.view(num_iwae, -1), dim=0)
      logqz = log_mean_exp(logqz.view(num_iwae, -1), dim=0)
      likelihood = log_mean_exp(likelihood.view(num_iwae, -1), dim=0)
    elbo = torch.mean(elbo)
    logpz = torch.mean(logpz)
    logqz = torch.mean(logqz)
    likelihood = torch.mean(likelihood)

    output_dict = {
        "inputs": x,
        "logits": x_mean.reshape(-1, 1, 28, 28),
        "mean": mu,
        "log_var": logvar
    }
    return elbo, output_dict


# ==============================
# === Hyper Linear-VAE Class ===
# ==============================


def replace_module(model: nn.Module, hyper_config: HyperConfig) -> None:
  for name, module in model.named_children():
    if len(list(module.children())) > 0:
      replace_module(module, hyper_config)

    if isinstance(module, nn.Linear):
      hyper_module = HyperLinear(module, hyper_config)
      setattr(model, name, hyper_module)

    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
      hyper_module = HyperModule(module, hyper_config)
      setattr(model, name, hyper_module)


class LinearHyperVae(LinearVae):

  def __init__(self, encoder, decoder, hyper_config):
    super().__init__(encoder, decoder)

    self.encoder = encoder
    self.decoder = decoder

    replace_module(self.encoder, hyper_config)
    replace_module(self.decoder, hyper_config)
    self.hyper_config = hyper_config

    self._hyper_modules = []
    self._register_hyper_modules()

  def _register_hyper_modules(self):
    for module in self.encoder.modules():
      if isinstance(module, HyperModule):
        self._hyper_modules.append(module)

    for module in self.decoder.modules():
      if isinstance(module, HyperModule):
        self._hyper_modules.append(module)

  def set_beta(self, beta: torch.Tensor) -> None:
    for hm in self._hyper_modules:
      hm.set_beta(beta)

  def reset_beta(self) -> None:
    for hm in self._hyper_modules:
      hm.reset_beta()

  def sample_beta(self, x: torch.Tensor, warmup=False):
    batch_size = x.shape[0]
    device = x.device
    a = self.hyper_config.sample_range[0]
    b = self.hyper_config.sample_range[1]
    sample_dict = {}

    if self.hyper_config.sample_type == "log_uniform":
      a = math.log(a)
      b = math.log(b)
      sample_dict["net_beta"] = torch.FloatTensor(batch_size,
                                                  1).uniform_(a, b).to(device)
      sample_dict["trans_beta"] = torch.exp(sample_dict["net_beta"])

    elif self.hyper_config.sample_type == "fixed_log_uniform":
      cons = math.sqrt(3)
      sample_dict["net_beta"] = torch.FloatTensor(batch_size,
                                                  1).uniform_(-cons,
                                                              cons).to(device)
      # Equivalent to setting a = -3 and b = 1
      norm_beta = sample_dict["net_beta"] * (2 * cons) / 3
      sample_dict["trans_beta"] = torch.pow(10, norm_beta - 1)

    elif self.hyper_config.sample_type == "fixed_normal":
      sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).normal_(
          mean=0, std=1).to(device)
      sample_dict["trans_beta"] = stretch_sigmoid(
          sample_dict["net_beta"] - 2, low=1e-3, high=10, slope=2)

    elif self.hyper_config.sample_type == "uniform":
      sample_dict["net_beta"] = torch.FloatTensor(batch_size,
                                                  1).uniform_(a, b).to(device)
      sample_dict["trans_beta"] = torch.exp(sample_dict["net_beta"])

    else:
      raise NotImplementedError

    return sample_dict

  def fixed_beta(self, x: torch.Tensor, beta: float):
    batch_size = x.shape[0]
    device = x.device
    ones = torch.ones(batch_size, 1).to(device)

    if self.hyper_config.sample_type == "log_uniform":
      beta = ones * beta
      trans_beta = torch.log(beta)

    elif self.hyper_config.sample_type == "fixed_log_uniform":
      cons = math.sqrt(3)
      beta = (ones * beta) * 3 / (2 * cons)
      trans_beta = torch.log10(beta) + 1

    elif self.hyper_config.sample_type == "fixed_normal":
      beta = ones * beta
      trans_beta = stretch_sigmoid_inv(beta, low=1e-3, high=10, slope=2) + 2

    elif self.hyper_config.sample_type == "normal":
      # sample_dict["net_beta"] = torch.FloatTensor(batch_size, 1).normal_(0, std=1).to(device)
      # sample_dict["trans_beta"] = stretch_sigmoid(sample_dict["net_beta"], low=a, high=b)
      trans_beta = None

    elif self.hyper_config.sample_type == "uniform":
      trans_beta = ones * math.log(beta)

    else:
      raise NotImplementedError

    return trans_beta

  def sample_forward(self, x):
    sample_dict = self.sample_beta(x)
    self.set_beta(sample_dict["net_beta"])
    elbo, output_dict = self.forward(x)
    output_dict["beta"] = sample_dict["trans_beta"]
    return output_dict

  def fixed_forward(self, x, beta):
    fixed_beta = self.fixed_beta(x, beta)
    self.set_beta(fixed_beta)
    elbo, output_dict = self.forward(x)

    batch_size = x.shape[0]
    device = x.device
    output_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta
    return output_dict

  def hyper_ignore_forward(self, x):
    zero_beta = self.fixed_beta(x, 0)
    self.reset_beta()
    elbo, output_dict = self.forward(x)
    output_dict["beta"] = torch.zeros(zero_beta.shape)
    return output_dict
