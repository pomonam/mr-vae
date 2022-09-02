import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from experiments.misc.td_vae.input_pipeline import load_data
from experiments.misc.td_vae.model import Encoder, Decoder
from experiments.train_utils import evaluate
from experiments.train_utils import predict
from experiments.train_utils import train
from experiments.wandb_utils import init_wandb
from src.config import TrainConfig
from src.models.beta_tc_vae import BetaTCVAE
from src.models.beta_vae import log_sum_exp
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default="hv_binary_image_debug")

parser.add_argument("--total_epochs", type=int, default=10)

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--beta", type=float, default=1.)
parser.add_argument("--schedule", type=str, default="monotonic")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_final_checkpoint", type=int, default=0)
parser.add_argument("--save_freq", type=int, default=50)
parser.add_argument("--eval_freq", type=int, default=10)
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


class Criterion(nn.Module):

  @staticmethod
  def _compute_log_gauss_density(z, mu, log_var):
    """element-wise computation"""
    return -0.5 * (
        torch.log(torch.tensor([2 * np.pi]).to(z.device))
        + log_var
        + (z - mu) ** 2 * torch.exp(-log_var)
    )

  @staticmethod
  def _log_importance_weight_matrix(batch_size, dataset_size):
    """Compute importance weigth matrix for MSS
    Code from (https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py)
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[:: M + 1] = 1 / N
    W.view(-1)[1:: M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()

  @staticmethod
  def get_metric_lst():
    return ["loss", "rate", "distortion"]

  @staticmethod
  def get_eval_metric_lst():
    return Criterion.get_metric_lst() + ["mi"]

  @staticmethod
  def forward(recon_x, x, mu, log_var, z, beta):
    dataset_size = 400
    recon_loss = F.mse_loss(
      recon_x.reshape(x.shape[0], -1),
      x.reshape(x.shape[0], -1),
      reduction="none",
    ).sum(dim=-1)

    log_q_z_given_x = Criterion._compute_log_gauss_density(z, mu, log_var).sum(
      dim=-1
    )  # [B]

    log_prior = Criterion._compute_log_gauss_density(
      z, torch.zeros_like(z), torch.zeros_like(z)
    ).sum(
      dim=-1
    )  # [B]

    log_q_batch_perm = Criterion._compute_log_gauss_density(
      z.reshape(z.shape[0], 1, -1),
      mu.reshape(1, z.shape[0], -1),
      log_var.reshape(1, z.shape[0], -1),
    )  # [B x B x Latent_dim]

    log_q_z = torch.logsumexp(log_q_batch_perm.sum(dim=-1), dim=-1) - torch.log(
      torch.tensor([z.shape[0] * dataset_size]).to(z.device)
    )  # MWS [B]
    log_prod_q_z = (
        torch.logsumexp(log_q_batch_perm, dim=1)
        - torch.log(torch.tensor([z.shape[0] * dataset_size]).to(z.device))
    ).sum(
      dim=-1
    )  # MWS [B]

    mutual_info_loss = log_q_z_given_x - log_q_z
    TC_loss = log_q_z - log_prod_q_z
    dimension_wise_KL = log_prod_q_z - log_prior

    # TODO(JB): Need to find the correct alpha, gamma configuration ...
    alpha = 1
    gamma = 1
    loss_dict = {
        "loss": (recon_loss
                + alpha * mutual_info_loss
                + beta * TC_loss
                + gamma * dimension_wise_KL).mean(dim=0),
        "distortion": recon_loss.mean(dim=0),
        "rate": (alpha * mutual_info_loss
                + beta * TC_loss
                + gamma * dimension_wise_KL).mean(dim=0)
    }
    return loss_dict

  @staticmethod
  def eval_forward(recon_x, x, mu, log_var, z, beta):
    recon_loss = F.binary_cross_entropy(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
    ).sum(dim=-1)

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    # Compute MI as well for eval_forward.
    batch_size, nz = mu.size()
    neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 *
                   (1 + log_var).sum(-1)).mean()
    z, mu, log_var = z.unsqueeze(1), mu.unsqueeze(1), log_var.unsqueeze(1)
    var = log_var.exp()
    dev = z - mu
    log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                  0.5 * (nz * math.log(2 * math.pi) + log_var.sum(-1))
    log_qz = log_sum_exp(log_density, dim=1) - math.log(batch_size)
    mi = neg_entropy - log_qz.mean(-1)

    loss_dict = {
        "loss": (recon_loss + beta * kld).mean(dim=0),
        "distortion": recon_loss.mean(dim=0),
        "rate": kld.mean(dim=0),
        "mi": mi
    }
    return loss_dict


def build_criterion(device):
  loss_fnc = Criterion()
  return loss_fnc.to(device)


def build_model(device):
  encoder = Encoder()
  decoder = Decoder()

  model = BetaTCVAE(
      encoder=encoder,
      decoder=decoder,
  )
  model.reconstruction_loss = "bce"
  return model.to(device)


def main():
  init_wandb(
      args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))
  cfg = TrainConfig(args)

  seed_everything(cfg.seed)
  model = build_model(DEVICE)

  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-4)
  criterion = build_criterion(DEVICE)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[1000], gamma=10 ** (-1 / 7)
  )

  train_loader = load_data(
      "train", cfg.batch_size, workers=2, data_path="../../../logs/data")
  test_loader = load_data(
      "test", cfg.batch_size, workers=2, data_path="../../../logs/data")

  train(model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        DEVICE,
        cfg)
  evaluate(model,
           train_loader,
           criterion,
           cfg.total_epochs,
           "train_eval",
           DEVICE)
  evaluate(model, test_loader, criterion, cfg.total_epochs, "test", DEVICE)

  true_data, reconstructions, generations = predict(model, test_loader, DEVICE)
  column_names = ["images_id", "truth", "reconstruction", "normal_generation"]
  data_to_log = []
  for i in range(len(true_data)):
    data_to_log.append([
        f"img_{i}",
        wandb.Image(np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1)),
        wandb.Image(
            np.clip(
                np.moveaxis(reconstructions[i].cpu().detach().numpy(), 0, -1),
                0,
                255.0,
            )),
        wandb.Image(
            np.clip(
                np.moveaxis(generations[i].cpu().detach().numpy(), 0, -1),
                0,
                255.0,
            )),
    ])
  val_table = wandb.Table(data=data_to_log, columns=column_names)
  wandb.log({"image": val_table})

  if args.save_final_checkpoint:
    save_checkpoint = \
      os.path.join("checkpoints", "base_{}_{}_{}_{}_{}.pth".
                   format(args.data_name, args.encoder_name,
                          args.decoder_name, args.beta, args.schedule))
    log_info = {
        "state_dict": model.state_dict(),
    }
    torch.save(log_info, save_checkpoint)

  wandb.finish()


if __name__ == "__main__":
  main()
