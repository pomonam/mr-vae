import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from experiments.text.input_pipeline import load_data
from experiments.text.models import LstmEncoder
from experiments.text.models import LstmDecoder
from experiments.train_utils import evaluate
from experiments.train_utils import train
from experiments.wandb_utils import init_wandb
from src.config import TrainConfig
from src.models.beta_vae import BetaVAE
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default="hypervae-text-train")

parser.add_argument("--data_name", type=str, default="yahoo")

parser.add_argument("--total_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--beta", type=float, default=1.)
parser.add_argument("--schedule", type=str, default="constant")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_final_checkpoint", type=int, default=0)
parser.add_argument("--save_freq", type=int, default=500)
parser.add_argument("--eval_freq", type=int, default=50)
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


class TextCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

    @staticmethod
    def forward(recon_x, x, mu, log_var, z, beta):
        recon_loss = recon_x
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        loss_dict = {
            "loss": (recon_loss + beta * kld).mean(dim=0),
            "distortion": recon_loss.mean(dim=0),
            "rate": kld.mean(dim=0)
        }
        return loss_dict


def build_criterion(device):
    loss_fnc = TextCriterion()
    return loss_fnc.to(device)


def build_model(device):
    model = BetaVAE(
        encoder=LstmEncoder(),
        decoder=LstmDecoder(),
    )
    return model.to(device)


def main():
    init_wandb(
        args.checkpoint_dir,
        project_name=args.experiment_name,
        config=vars(args))
    cfg = TrainConfig(args)

    seed_everything(cfg.seed)
    model = build_model(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    criterion = build_criterion(DEVICE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5)

    train_loader = load_data(
        args.data_name,
        "train",
        cfg.batch_size,
        data_path="../../logs/text_data")
    test_loader = load_data(
        args.data_name,
        "test",
        cfg.batch_size,
        data_path="../../logs/text_data")

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

    if args.save_final_checkpoint is not None:
        save_checkpoint = \
          os.path.join("checkpoints", "base_{}_{}.pth".format(args.beta, args.schedule))
        log_info = {
            "state_dict": model.state_dict(),
        }
        torch.save(log_info, save_checkpoint)

    wandb.finish()


if __name__ == "__main__":
    main()
