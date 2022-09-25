import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from experiments.hyper_train_utils import hyper_evaluate
from experiments.hyper_train_utils import hyper_predict
from experiments.hyper_train_utils import hyper_train
from alternative_experiments.image.hyper_models import HyperCelebConvDecoder
from alternative_experiments.image.hyper_models import HyperCelebConvEncoder
from alternative_experiments.image.hyper_models import HyperCelebResNetDecoder
from alternative_experiments.image.hyper_models import HyperCelebResNetEncoder
from alternative_experiments.image.hyper_models import HyperCifarConvDecoder
from alternative_experiments.image.hyper_models import HyperCifarConvEncoder
from alternative_experiments.image.hyper_models import HyperCifarResNetDecoder
from alternative_experiments.image.hyper_models import HyperCifarResNetEncoder
from experiments.image.input_pipeline import load_data
from experiments.wandb_utils import init_wandb
from src.config import HyperConfig
from src.config import TrainConfig
from src.hyper.beta_vae import HyperBetaVAE
from src.utils import log_sum_exp
from src.utils import seed_everything

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


class HyperImageCriterion(nn.Module):

  @staticmethod
  def get_metric_lst():
    return ["loss", "rate", "distortion"]

  @staticmethod
  def get_eval_metric_lst():
    return HyperImageCriterion.get_metric_lst() + ["mi"]

  @staticmethod
  def forward(recon_x, x, mu, log_var, z, beta):
    recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
    ).sum(dim=-1)

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    loss_dict = {
        "loss": (recon_loss + beta.squeeze(-1) * kld).mean(dim=0),
        "distortion": recon_loss.mean(dim=0),
        "rate": kld.mean(dim=0)
    }
    return loss_dict

  @staticmethod
  def eval_forward(recon_x, x, mu, log_var, z, beta):
    recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
    ).sum(dim=-1)

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    # Compute MI as well for eval forward.
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
  loss_fnc = HyperImageCriterion()
  return loss_fnc.to(device)


def build_model(data_name, arch_name, hyper_cfg, device):
  if data_name in ["cifar", "svhn"]:
    if data_name == "cifar":
      latent_dim = 256
    else:
      latent_dim = 32
    model = HyperBetaVAE(
        encoder=HyperCifarConvEncoder(latent_dim) if arch_name
        == "conv" else HyperCifarResNetEncoder(latent_dim),
        decoder=HyperCifarConvDecoder(latent_dim) if arch_name
        == "conv" else HyperCifarResNetDecoder(latent_dim),
        hyper_cfg=hyper_cfg)
  else:
    model = HyperBetaVAE(
        encoder=HyperCelebConvEncoder()
        if arch_name == "conv" else HyperCelebResNetEncoder(),
        decoder=HyperCelebConvDecoder()
        if arch_name == "conv" else HyperCelebResNetDecoder(),
        hyper_cfg=hyper_cfg)
  model.reconstruction_loss = "mse"
  return model.to(device)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment_name", type=str, default="hvae_image_debug")

  parser.add_argument("--data_name", type=str, default="cifar")
  parser.add_argument("--arch_name", type=str, default="resnet")

  parser.add_argument("--hyper_config_summary", type=str, default="linear_default")

  parser.add_argument("--total_epochs", type=int, default=10)
  parser.add_argument("--warmup_epochs", type=int, default=10)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--batch_size", type=int, default=128)

  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--checkpoint_dir", type=str, default=None)
  parser.add_argument("--save_final_checkpoint", type=int, default=1)
  parser.add_argument("--save_freq", type=int, default=50)
  # Never evaluate during training.
  parser.add_argument("--eval_freq", type=int, default=2000)
  args = parser.parse_args()

  init_wandb(
      args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))
  cfg = TrainConfig(args)
  hyper_cfg = HyperConfig(args)

  seed_everything(cfg.seed)
  model = build_model(args.data_name, args.arch_name, hyper_cfg, DEVICE)
  print(model)

  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
  criterion = build_criterion(DEVICE)

  scheduler1 = torch.optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=1e-10,
      end_factor=1.,
      total_iters=cfg.warmup_epochs)
  cosine_epochs = max(cfg.total_epochs - cfg.warmup_epochs, 1)
  scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=cosine_epochs, eta_min=1e-6)
  scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[scheduler1, scheduler2],
      milestones=[cfg.warmup_epochs])

  train_loader = load_data(
      args.data_name,
      "train",
      cfg.batch_size,
      workers=4,
      data_path="../../logs/data")
  test_loader = load_data(
      args.data_name,
      "test",
      cfg.batch_size,
      workers=4,
      data_path="../../logs/data")

  hyper_train(model,
              train_loader,
              test_loader,
              criterion,
              optimizer,
              scheduler,
              DEVICE,
              cfg,
              hyper_cfg)
  hyper_evaluate(
      model,
      train_loader,
      criterion,
      cfg.total_epochs,
      "train_eval",
      hyper_cfg,
      DEVICE,
      train_loader=train_loader)
  hyper_evaluate(
      model,
      test_loader,
      criterion,
      cfg.total_epochs,
      "test",
      hyper_cfg,
      DEVICE,
      train_loader=train_loader)

  for sample in model.get_log_uniform_samples(4):
    true_data, reconstructions, generations = hyper_predict(model, test_loader, sample, DEVICE)
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
    wandb.log({"image_at_{}".format(sample): val_table})

  if args.save_final_checkpoint:
    save_checkpoint = \
      os.path.join("checkpoints", "hyper_{}_{}_seed{}.pth".format(args.data_name,
                                                                  args.arch_name,
                                                                  args.seed))
    log_info = {
        "state_dict": model.state_dict(),
    }
    torch.save(log_info, save_checkpoint)

  wandb.finish()


if __name__ == "__main__":
  main()
