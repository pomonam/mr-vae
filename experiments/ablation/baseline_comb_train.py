import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from experiments.binary_image.input_pipeline import load_mnist_data
from experiments.binary_image.input_pipeline import load_omniglot_data
from experiments.binary_image.models import ConvDecoder
from experiments.binary_image.models import ConvEncoder
from experiments.binary_image.models import ResNetDecoder
from experiments.binary_image.models import ResNetEncoder
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


class BinaryImageCriterion(nn.Module):

  @staticmethod
  def get_metric_lst():
    return ["loss", "rate", "distortion"]

  @staticmethod
  def get_eval_metric_lst():
    return BinaryImageCriterion.get_metric_lst() + ["mi"]

  @staticmethod
  def forward(recon_x, x, mu, log_var, z, beta):
    recon_loss = F.binary_cross_entropy(
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
  loss_fnc = BinaryImageCriterion()
  return loss_fnc.to(device)


def build_model(encoder_name, decoder_name, device):
  if encoder_name == "conv":
    encoder = ConvEncoder()
  elif encoder_name == "resnet":
    encoder = ResNetEncoder()
  else:
    raise

  if decoder_name == "conv":
    decoder = ConvDecoder()
  elif decoder_name == "resnet":
    decoder = ResNetDecoder()
  else:
    raise

  model = BetaVAE(
      encoder=encoder,
      decoder=decoder,
  )
  model.reconstruction_loss = "bce"
  return model.to(device)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--experiment_name", type=str, default="hvae_bimage_debug")

  parser.add_argument("--data_name", type=str, default="mnist")
  parser.add_argument("--encoder_name", type=str, default="conv")
  parser.add_argument("--decoder_name", type=str, default="conv")

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
  model = build_model(args.encoder_name, args.decoder_name, DEVICE)
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

  if args.data_name == "mnist":
    train_loader = load_mnist_data(
        "train", cfg.batch_size, workers=2, data_path="../../logs/data")
    test_loader = load_mnist_data(
        "test", cfg.batch_size, workers=2, data_path="../../logs/data")
  elif args.data_name == "omniglot":
    train_loader = load_omniglot_data(
        "train", cfg.batch_size, workers=2, data_path="../../logs/data")
    test_loader = load_omniglot_data(
        "test", cfg.batch_size, workers=2, data_path="../../logs/data")
  else:
    raise NotImplementedError

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

  # Uncomment ot compute NLL.
  # TODO(JB): This function gives negative NLL?
  # test_loader = load_mnist_data(
  #   "test", 10000, workers=0, data_path="../../logs/data")
  # test_data = next(iter(test_loader))
  # test_data = test_data[0]
  # test_data = test_data.to(DEVICE).type(torch.float)
  # with torch.no_grad():
  #   nll = []
  #   for i in range(5):
  #     nll_i = model.get_nll(test_data, n_samples=500, batch_size=500)
  #     print(f"Round {i + 1} nll: {nll_i}")
  #     nll.append(nll_i)
  # wandb.log({"nll_mean": np.mean(nll), "nll_std": np.std(nll)})

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
