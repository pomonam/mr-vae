import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from experiments.image.input_pipeline import load_data
from experiments.image.models import CelebConvDecoder
from experiments.image.models import CelebConvEncoder
from experiments.image.models import CelebResNetDecoder
from experiments.image.models import CelebResNetEncoder
from experiments.image.models import CifarConvDecoder
from experiments.image.models import CifarConvEncoder
from experiments.image.models import CifarResNetDecoder
from experiments.image.models import CifarResNetEncoder
from experiments.train_utils import evaluate
from experiments.train_utils import predict
from experiments.train_utils import train
from experiments.wandb_utils import init_wandb
from src.config import TrainConfig
from src.models.beta_vae import BetaVAE
from src.utils import log_sum_exp
from src.utils import seed_everything

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


class ImageCriterion(nn.Module):

  @staticmethod
  def get_metric_lst():
    return ["loss", "rate", "distortion"]

  @staticmethod
  def get_eval_metric_lst():
    return ImageCriterion.get_metric_lst() + ["mi"]

  @staticmethod
  def forward(recon_x, x, mu, log_var, z, beta):
    recon_loss = F.mse_loss(
        recon_x.reshape(x.shape[0], -1),
        x.reshape(x.shape[0], -1),
        reduction="none",
    ).sum(dim=-1)

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    loss_dict = {
        "loss": (recon_loss + beta * kld).mean(dim=0),
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
  loss_fnc = ImageCriterion()
  return loss_fnc.to(device)


def build_model(data_name, arch_name, device):
  if data_name in ["cifar", "svhn"]:
    if data_name == "cifar":
      latent_dim = 256
    else:
      latent_dim = 32
    model = BetaVAE(
        encoder=CifarConvEncoder(latent_dim)
        if arch_name == "conv" else CifarResNetEncoder(latent_dim),
        decoder=CifarConvDecoder(latent_dim)
        if arch_name == "conv" else CifarResNetDecoder(latent_dim),
    )
  else:
    model = BetaVAE(
        encoder=CelebConvEncoder()
        if arch_name == "conv" else CelebResNetEncoder(),
        decoder=CelebConvDecoder()
        if arch_name == "conv" else CelebResNetDecoder(),
    )
  model.reconstruction_loss = "mse"
  return model.to(device)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--experiment_name", type=str, default="hvae_image_debug")

  parser.add_argument("--data_name", type=str, default="cifar")
  parser.add_argument("--arch_name", type=str, default="resnet")

  parser.add_argument("--total_epochs", type=int, default=10)
  parser.add_argument("--warmup_epochs", type=int, default=10)

  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--beta", type=float, default=1.)
  parser.add_argument("--schedule", type=str, default="monotonic")

  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--checkpoint_dir", type=str, default=None)
  parser.add_argument("--save_final_checkpoint", type=int, default=0)
  parser.add_argument("--save_freq", type=int, default=50)
  parser.add_argument("--eval_freq", type=int, default=10)
  args = parser.parse_args()

  init_wandb(
      args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))
  cfg = TrainConfig(args)

  seed_everything(cfg.seed)
  model = build_model(args.data_name, args.arch_name, DEVICE)
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
      optimizer, T_max=cosine_epochs)
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

  if args.save_final_checkpoint is not None:
    save_checkpoint = \
      os.path.join("checkpoints", "base_{}_{}_{}_{}.pth".format(args.data_name, args.arch_name,
                                                                args.beta, args.schedule))
    log_info = {
        "state_dict": model.state_dict(),
    }
    torch.save(log_info, save_checkpoint)

  wandb.finish()


if __name__ == "__main__":
  main()
