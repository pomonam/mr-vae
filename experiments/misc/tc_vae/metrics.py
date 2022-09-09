""" Code adapted from:
- https://github.com/rtqichen/beta-tcvae/blob/master/metric_helpers/mi_metric.py
- https://github.com/rtqichen/beta-tcvae/blob/master/disentanglement_metrics.py
"""

import math
import os

import torch
from torch.autograd import Variable
from tqdm import tqdm

from experiments.misc.tc_vae.dist import Normal


def MIG(mi_normed):
  return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_shapes(marginal_entropies, cond_entropies):
  factor_entropies = [6, 40, 32, 32]
  mutual_infos = marginal_entropies[None] - cond_entropies
  mutual_infos = torch.sort(
      mutual_infos, dim=1, descending=True)[0].clamp(min=0)
  mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
  metric = MIG(mi_normed)
  return metric


def estimate_entropies(qz_samples, qz_params, n_samples=10000, weights=None):
  q_dist = Normal()

  # Only take a sample subset of the samples
  if weights is None:
    qz_samples = qz_samples.index_select(
        1, Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda()))
  else:
    sample_inds = torch.multinomial(weights, n_samples, replacement=True)
    qz_samples = qz_samples.index_select(1, sample_inds)

  K, S = qz_samples.size()
  N, _, nparams = qz_params.size()
  assert (nparams == q_dist.nparams)
  assert (K == qz_params.size(1))

  if weights is None:
    weights = -math.log(N)
  else:
    weights = torch.log(weights.view(N, 1, 1) / weights.sum())

  entropies = torch.zeros(K).cuda()

  pbar = tqdm(total=S)
  k = 0
  while k < S:
    batch_size = min(10, S - k)
    logqz_i = q_dist.log_density(
        qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
        qz_params.view(N, K, 1, nparams).expand(N, K, S,
                                                nparams)[:, :,
                                                         k:k + batch_size])
    k += batch_size

    # computes - log q(z_i) summed over minibatch
    entropies += -torch.logsumexp(
        logqz_i + weights, dim=0, keepdim=False).data.sum(1)
    pbar.update(batch_size)
  pbar.close()

  entropies /= S

  return entropies


def mutual_info_metric_shapes(model, loader):
  N = len(loader.dataset)
  K = 10
  nparams = 2
  model.eval()

  print('Computing q(z|x) distributions.')
  qz_params = torch.Tensor(N, K, nparams)

  n = 0
  for xs in loader:
    xs = xs[0]
    batch_size = xs.size(0)
    xs = Variable(xs.view(batch_size, 1, 64, 64).cuda(), volatile=True)
    outputs = model.encoder.forward(xs)
    qz_params[n:n + batch_size] = torch.cat(
        (outputs["embedding"].unsqueeze(-1),
         outputs["log_covariance"].unsqueeze(-1)),
        -1).data
    n += batch_size

  qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda())
  # qz_samples = model.q_dist.sample(params=qz_params)
  mu = qz_params.select(-1, 0)
  log_sigma = qz_params.select(-1, 1)
  qz_samples = model._sample_gauss(mu, log_sigma)[0]

  print('Estimating marginal entropies.')
  # marginal entropies
  marginal_entropies = estimate_entropies(
      qz_samples.view(N, K).transpose(0, 1), qz_params.view(N, K, nparams))

  marginal_entropies = marginal_entropies.cpu()
  cond_entropies = torch.zeros(4, K)

  print('Estimating conditional entropies for scale.')
  for i in range(6):
    qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
    qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 6, K).transpose(0, 1),
        qz_params_scale.view(N // 6, K, nparams))

    cond_entropies[0] += cond_entropies_i.cpu() / 6

  print('Estimating conditional entropies for orientation.')
  for i in range(40):
    qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
    qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 40, K).transpose(0, 1),
        qz_params_scale.view(N // 40, K, nparams))

    cond_entropies[1] += cond_entropies_i.cpu() / 40

  print('Estimating conditional entropies for pos x.')
  for i in range(32):
    qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
    qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 32, K).transpose(0, 1),
        qz_params_scale.view(N // 32, K, nparams))

    cond_entropies[2] += cond_entropies_i.cpu() / 32

  print('Estimating conditional entropies for pox y.')
  for i in range(32):
    qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
    qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 32, K).transpose(0, 1),
        qz_params_scale.view(N // 32, K, nparams))

    cond_entropies[3] += cond_entropies_i.cpu() / 32

  metric = compute_metric_shapes(marginal_entropies, cond_entropies)
  return metric, marginal_entropies, cond_entropies
