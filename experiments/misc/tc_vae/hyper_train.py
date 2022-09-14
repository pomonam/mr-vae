import os
import time
import math
from numbers import Number
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from experiments.wandb_utils import init_wandb
import experiments.misc.tc_vae.lib.dist as dist
import experiments.misc.tc_vae.lib.utils as utils
import experiments.misc.tc_vae.lib.datasets as dset
from experiments.misc.tc_vae.metrics import mutual_info_metric_shapes
from experiments.misc.tc_vae.elbo_decomposition import elbo_decomposition
from src.hyper.layers import get_hyper_layer
from src.utils import seed_everything
import wandb
from src.hyper.base_model import BaseHyperEncoder, BaseHyperDecoder
from src.config import HyperConfig

_SQRT3 = math.sqrt(3)
_LOG_A = math.log(0.001)
_LOG_RED_A = math.log(1)
_LOG_B = math.log(10)
_LOG_M = (_LOG_A + _LOG_B) / 2
_LOG_RED_M = (_LOG_RED_A + _LOG_B) / 2
_LOG_DIFF = _LOG_M - _LOG_A
_LOG_RED_DIFF = _LOG_RED_M - _LOG_RED_A

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


class HyperMLPEncoder(BaseHyperEncoder):

  def __init__(self, output_dim, hyper_cfg):
    super(HyperMLPEncoder, self).__init__()
    self.output_dim = output_dim

    self.fc1 = nn.Linear(4096, 1200)
    self.hyper_fc1 = get_hyper_layer(1200, hyper_cfg)
    self.fc2 = nn.Linear(1200, 1200)
    self.hyper_fc2 = get_hyper_layer(1200, hyper_cfg)
    self.fc3 = nn.Linear(1200, output_dim)
    self.act = nn.ReLU(inplace=True)

  def forward(self, x):
    h = x.view(-1, 64 * 64)
    h = self.act(self.fc1(h))
    h = self.hyper_fc1(h)
    h = self.act(self.fc2(h))
    h = self.hyper_fc2(h)
    h = self.fc3(h)
    h = self.hyper_fc3(h)
    z = h.view(x.size(0), self.output_dim)
    return z


class HyperMLPDecoder(BaseHyperDecoder):

  def __init__(self, input_dim, hyper_cfg):
    super(HyperMLPDecoder, self).__init__()
    self.net = nn.Sequential(
        nn.Linear(input_dim, 1200),
        nn.Tanh(),
        get_hyper_layer(1200, hyper_cfg),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        get_hyper_layer(1200, hyper_cfg),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        get_hyper_layer(1200, hyper_cfg),
        nn.Linear(1200, 4096)
    )

  def forward(self, z):
    h = z.view(z.size(0), -1)
    h = self.net(h)
    mu_img = h.view(z.size(0), 1, 64, 64)
    return mu_img


class VAE(nn.Module):

  def __init__(self,
               z_dim,
               hyper_cfg,
               use_cuda=False,
               prior_dist=dist.Normal(),
               q_dist=dist.Normal(),
               include_mutinfo=True,
               tcvae=False,
               conv=False,
               mss=False):
    super(VAE, self).__init__()

    self.use_cuda = use_cuda
    self.z_dim = z_dim
    self.hyper_cfg = hyper_cfg
    self.include_mutinfo = include_mutinfo
    self.tcvae = tcvae
    self.lamb = 0
    # self.beta = 1
    self.mss = mss
    self.x_dist = dist.Bernoulli()

    # Model-specific
    # distribution family of p(z)
    self.prior_dist = prior_dist
    self.q_dist = q_dist
    # hyperparameters for prior p(z)
    self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

    # create the encoder and decoder networks
    self.encoder = HyperMLPEncoder(z_dim * self.q_dist.nparams, hyper_cfg)
    self.decoder = HyperMLPDecoder(z_dim, hyper_cfg)

  def set_inputs_for_net(self, x, value):
    sample_dict = self.sample_inverse(x, value)
    self.set_net_inputs(sample_dict["net"])

  def sample(self, x: torch.Tensor) -> dict:
    try:
      batch_size = x.shape[0]
      device = x.device
    except AttributeError:
      # This is for text models.
      batch_size = x.batch_size
      device = x._batch["text_ids"].device

    sample_dict = {}
    sample_dict["net"] = (
        torch.FloatTensor(batch_size, 1).uniform_(-_SQRT3, _SQRT3).to(device))
    beta = sample_dict["net"] * (_SQRT3 / 3)
    beta = beta * _LOG_RED_DIFF + _LOG_RED_M
    sample_dict["beta"] = torch.exp(beta)
    return sample_dict

  def sample_inverse(self, x: torch.Tensor, value: float) -> dict:
    try:
      batch_size = x.shape[0]
      device = x.device
    except AttributeError:
      # This is for text models.
      batch_size = x.batch_size
      device = x._batch["text_ids"].device

    sample_dict = {}
    ones = torch.ones(batch_size, 1).to(device)
    beta = value * ones
    sample_dict["beta"] = torch.ones(batch_size, 1).to(device) * beta
    net_beta = (torch.log(sample_dict["beta"]) - _LOG_RED_M) / _LOG_RED_DIFF
    sample_dict["net"] = net_beta * (3 / _SQRT3)
    return sample_dict

  def _get_prior_params(self, batch_size=1):
    expanded_size = (batch_size,) + self.prior_params.size()
    prior_params = Variable(self.prior_params.expand(expanded_size))
    return prior_params

  # def model_sample(self, value, batch_size=1):
  #   sample_dict = self.sample_inverse(torch.Tensor(1, 1).to(DEVICE), value)
  #   self.set_net_inputs(sample_dict["net"])
  #   # sample from prior (value will be sampled by guide when computing the ELBO)
  #   prior_params = self._get_prior_params(batch_size)
  #   zs = self.prior_dist.sample(params=prior_params)
  #   # decode the latent code z
  #   x_params = self.decoder.forward(zs)
  #   return x_params

  def encode(self, x):
    x = x.view(x.size(0), 1, 64, 64)
    # net_inputs should be already set.
    z_params = self.encoder.forward(x).view(
        x.size(0), self.z_dim, self.q_dist.nparams)
    zs = self.q_dist.sample(params=z_params)
    return zs, z_params

  def decode(self, z):
    # net_inputs should be already set.
    x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
    xs = self.x_dist.sample(params=x_params)
    return xs, x_params

  def sample_reconstruct_img(self, x):
    sample_dict = self.sample(x)
    self.set_net_inputs(sample_dict["net"])
    zs, z_params = self.encode(x)
    xs, x_params = self.decode(zs)
    return xs, x_params, zs, z_params, sample_dict

  def set_net_inputs(self, value: torch.Tensor) -> None:
    self.encoder.set_net_inputs(value)
    self.decoder.set_net_inputs(value)

  def fixed_reconstruct_img(self, x, value):
    sample_dict = self.sample_inverse(x, value)
    self.set_net_inputs(sample_dict["net"])
    zs, z_params = self.encode(x)
    xs, x_params = self.decode(zs)
    return xs, x_params, zs, z_params

  def _log_importance_weight_matrix(self, batch_size, dataset_size):
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

  def sample_elbo(self, x, dataset_size):
    # log p(x|z) + log p(z) - log q(z|x)
    batch_size = x.size(0)
    x = x.view(batch_size, 1, 64, 64)
    prior_params = self._get_prior_params(batch_size)
    x_recon, x_params, zs, z_params, sample_dict = self.sample_reconstruct_img(x)
    logpx = self.x_dist.log_density(
        x, params=x_params).view(batch_size, -1).sum(1)
    logpz = self.prior_dist.log_density(
        zs, params=prior_params).view(batch_size, -1).sum(1)
    logqz_condx = self.q_dist.log_density(
        zs, params=z_params).view(batch_size, -1).sum(1)

    elbo = logpx + logpz - logqz_condx

    # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
    _logqz = self.q_dist.log_density(
        zs.view(batch_size, 1, self.z_dim),
        z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams))

    if not self.mss:
      # minibatch weighted sampling
      logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) -
                             math.log(batch_size * dataset_size)).sum(1)
      logqz = (
          logsumexp(_logqz.sum(2), dim=1, keepdim=False) -
          math.log(batch_size * dataset_size))
    else:
      # minibatch stratified sampling
      logiw_matrix = Variable(
          self._log_importance_weight_matrix(batch_size,
                                             dataset_size).type_as(_logqz.data))
      logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
      logqz_prodmarginals = logsumexp(
          logiw_matrix.view(batch_size, batch_size, 1) + _logqz,
          dim=1,
          keepdim=False).sum(1)

    if not self.tcvae:
      if self.include_mutinfo:
        modified_elbo = logpx - sample_dict["beta"].squeeze(-1) * ((logqz_condx - logpz) - self.lamb *
                                             (logqz_prodmarginals - logpz))
      else:
        modified_elbo = logpx - sample_dict["beta"].squeeze(-1) * ((logqz - logqz_prodmarginals) +
                                             (1 - self.lamb) *
                                             (logqz_prodmarginals - logpz))
    else:
      if self.include_mutinfo:
        modified_elbo = logpx - \
            (logqz_condx - logqz) - \
            sample_dict["beta"].squeeze(-1) * (logqz - logqz_prodmarginals) - \
            (1 - self.lamb) * (logqz_prodmarginals - logpz)
      else:
        modified_elbo = logpx - \
            sample_dict["beta"].squeeze(-1) * (logqz - logqz_prodmarginals) - \
            (1 - self.lamb) * (logqz_prodmarginals - logpz)

    return modified_elbo, elbo.detach()


def logsumexp(value, dim=None, keepdim=False):
  """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
  if dim is not None:
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
      m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
  else:
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    if isinstance(sum_exp, Number):
      return m + math.log(sum_exp)
    else:
      return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args):
  if args.dataset == 'shapes':
    train_set = dset.Shapes()
  elif args.dataset == 'faces':
    train_set = dset.Faces()
  else:
    raise ValueError('Unknown dataset ' + str(args.dataset))

  kwargs = {'num_workers': 4, 'pin_memory': True}
  train_loader = DataLoader(
      dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
  return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


# def display_samples(model, x, vis):
#   global win_samples, win_test_reco, win_latent_walk
#
#   # plot random samples
#   sample_mu = model.model_sample(batch_size=100).sigmoid()
#   sample_mu = sample_mu
#   images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
#   win_samples = vis.images(
#       images, 10, 2, opts={'caption': 'samples'}, win=win_samples)
#
#   # plot the reconstructed distribution for the first 50 test images
#   test_imgs = x[:50, :]
#   _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
#   reco_imgs = reco_imgs.sigmoid()
#   test_reco_imgs = torch.cat(
#       [test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)],
#       0).transpose(0, 1)
#   win_test_reco = vis.images(
#       list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()),
#       10,
#       2,
#       opts={'caption': 'test reconstruction image'},
#       win=win_test_reco)
#
#   # plot latent walks (change one variable while all others stay the same)
#   zs = zs[0:3]
#   batch_size, z_dim = zs.size()
#   xs = []
#   delta = torch.autograd.Variable(
#       torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
#   for i in range(z_dim):
#     vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(
#         7, z_dim).contiguous().type_as(zs)
#     vec[:, i] = 1
#     vec = vec * delta[:, None]
#     zs_delta = zs.clone().view(batch_size, 1, z_dim)
#     zs_delta[:, :, i] = 0
#     zs_walk = zs_delta + vec[None]
#     xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
#     xs.append(xs_walk)
#
#   xs = list(torch.cat(xs, 0).data.cpu())
#   win_latent_walk = vis.images(
#       xs, 7, 2, opts={'caption': 'latent walk'}, win=win_latent_walk)


def plot_elbo(train_elbo, vis):
  global win_train_elbo
  win_train_elbo = vis.line(
      torch.Tensor(train_elbo), opts={'markers': True}, win=win_train_elbo)


# def anneal_kl(args, vae, iteration):
#   if args.dataset == 'shapes':
#     warmup_iter = 7000
#   elif args.dataset == 'faces':
#     warmup_iter = 2500
#
#   if args.lambda_anneal:
#     vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
#   else:
#     vae.lamb = 0
#   if args.beta_anneal:
#     vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
#   else:
#     vae.beta = args.beta


def main():
  # parse command line arguments
  parser = argparse.ArgumentParser(description="parse args")
  parser.add_argument("--experiment_name", type=str, default="hvae_tcvae_debug")
  parser.add_argument("--hyper_config_summary", type=str, default="amlp_bn")

  parser.add_argument(
      '-d',
      '--dataset',
      default='shapes',
      type=str,
      help='dataset name',
      choices=['shapes', 'faces'])
  parser.add_argument(
      '-dist',
      default='normal',
      type=str,
      choices=['normal', 'laplace', 'flow'])
  parser.add_argument(
      '-n',
      '--num-epochs',
      default=50,
      type=int,
      help='number of training epochs')
  parser.add_argument(
      '-b', '--batch-size', default=2048, type=int, help='batch size')
  parser.add_argument(
      '-l', '--lr', default=1e-3, type=float, help='learning rate')
  parser.add_argument(
      '-z',
      '--latent-dim',
      default=10,
      type=int,
      help='size of latent dimension')
  parser.add_argument('--beta', default=6, type=float, help='ELBO penalty term')
  parser.add_argument('--tcvae', default=True, action='store_true')
  parser.add_argument('--exclude-mutinfo', action='store_true')
  parser.add_argument('--beta-anneal', action='store_true')
  parser.add_argument('--lambda-anneal', action='store_true')
  parser.add_argument(
      '--mss', action='store_true', help='use the improved minibatch estimator')
  parser.add_argument('--conv', action='store_true')
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--seed', type=int, default=1)
  # parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
  # parser.add_argument('--save', default='test1')
  parser.add_argument("--checkpoint_dir", type=str, default=None)
  parser.add_argument(
      '--log_freq', default=200, type=int, help='num iterations per log')
  args = parser.parse_args()

  hyper_cfg = HyperConfig(args)

  # torch.cuda.set_device(args.gpu)
  init_wandb(
      args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))

  seed_everything(args.seed)
  # data loader
  train_loader = setup_data_loaders(args)

  prior_dist = dist.Normal()
  q_dist = dist.Normal()

  vae = VAE(
      z_dim=args.latent_dim,
      hyper_cfg=hyper_cfg,
      use_cuda=True,
      prior_dist=prior_dist,
      q_dist=q_dist,
      include_mutinfo=not args.exclude_mutinfo,
      tcvae=args.tcvae,
      conv=args.conv,
      mss=args.mss)
  vae = vae.to(DEVICE)

  # setup the optimizer
  optimizer = optim.Adam(vae.parameters(), lr=args.lr)

  train_elbo = []

  # training loop
  dataset_size = len(train_loader.dataset)
  num_iterations = len(train_loader) * args.num_epochs
  iteration = 0
  # initialize loss accumulator
  elbo_running_mean = utils.RunningAverageMeter()
  while iteration < num_iterations:
    for i, x in enumerate(train_loader):
      iteration += 1
      batch_time = time.time()
      vae.train()
      # anneal_kl(args, vae, iteration)
      optimizer.zero_grad()
      # transfer to GPU
      x = x.to(DEVICE)

      x = Variable(x)
      # do ELBO gradient and accumulate loss
      obj, elbo = vae.sample_elbo(x, dataset_size)
      if utils.isnan(obj).any():
        raise ValueError('NaN spotted in objective.')
      obj.mean().mul(-1).backward()
      elbo_running_mean.update(elbo.mean().item())
      optimizer.step()

      # report training diagnostics
      if iteration % args.log_freq == 0:
        train_elbo.append(elbo_running_mean.avg)
        print(
            '[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)'
            % (iteration,
               time.time() - batch_time,
               1.,
               vae.lamb,
               elbo_running_mean.val,
               elbo_running_mean.avg))

        vae.eval()
        # utils.save_checkpoint({
        #     'state_dict': vae.state_dict(),
        #     'args': args}, args.save, 0)
        # eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset,
        #     os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)))

  # Report statistics after training
  vae.eval()
  # utils.save_checkpoint({
  #     'state_dict': vae.state_dict(),
  #     'args': args}, args.save, 0)
  # dataset_loader = DataLoader(
  #     train_loader.dataset, batch_size=1000, num_workers=0, shuffle=False)
  # logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
  #     elbo_decomposition(vae, dataset_loader)

  beta_lst = np.logspace(-2, 1, num=20, base=10)
  metrics_lst = []
  for beta in beta_lst:
    # Setting the hyper inputs.
    # vae.set_net_inputs(beta)
    metrics, _, _ = mutual_info_metric_shapes(vae, train_loader.dataset, hyper_mode=True, value=beta)
    metrics_lst.append(metrics)
    log_info = {
      "beta": beta,
      'metrics': metrics,
    }
    wandb.log(log_info)
  return vae


if __name__ == '__main__':
  model = main()
