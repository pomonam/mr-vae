import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from experiments.binary_image.input_pipeline import load_mnist_data
from experiments.binary_image.input_pipeline import load_omniglot_data
from experiments.binary_image.models import ResNetDecoder
from experiments.binary_image.models import ResNetEncoder
from experiments.train_utils import evaluate
from experiments.train_utils import predict
from experiments.train_utils import train
from experiments.wandb_utils import init_wandb
from src.config import TrainConfig
from src.models.beta_vae import BetaVAE
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default="hypervae-mnist-train")

parser.add_argument("--data_name", type=str, default="omniglot")
parser.add_argument("--encoder_name", type=str, default="cnn")
parser.add_argument("--decoder_name", type=str, default="cnn")

parser.add_argument("--total_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=128)
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


class BinaryImageCriterion(nn.Module):

    @staticmethod
    def get_metric_lst():
        return ["loss", "rate", "distortion"]

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


def build_criterion(device):
    loss_fnc = BinaryImageCriterion()
    return loss_fnc.to(device)


def build_model(encoder_name, decoder_name, device):
    model = BetaVAE(
        encoder=ResNetEncoder(),
        decoder=ResNetDecoder(),
    )
    return model.to(device)


def main():
    init_wandb(
        args.checkpoint_dir,
        project_name=args.experiment_name,
        config=vars(args))
    cfg = TrainConfig(args)

    seed_everything(cfg.seed)
    model = build_model(args.encoder_name, args.decoder_name, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = build_criterion(DEVICE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[200, 350, 500, 750], gamma=10**(-1 / 5))

    if args.data_name == "mnist":
        train_loader = load_mnist_data(
            "train", cfg.batch_size, workers=0, data_path="../../logs/data")
        # valid_loader = load_data("valid", cfg.batch_size, workers=0, data_path="../../logs/data")
        test_loader = load_mnist_data(
            "test", cfg.batch_size, workers=0, data_path="../../logs/data")
    elif args.data_name == "omniglot":
        train_loader = load_omniglot_data(
            "train", cfg.batch_size, workers=0, data_path="../../logs/")
        test_loader = load_omniglot_data(
            "test", cfg.batch_size, workers=0, data_path="../../logs/")
    elif args.data_name == "fashionmnist":
        train_loader = load_mnist_data(
            "train", cfg.batch_size, workers=0, data_path="../../logs/data")
        test_loader = load_mnist_data(
            "test", cfg.batch_size, workers=0, data_path="../../logs/data")
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
            wandb.Image(
                np.moveaxis(true_data[i].cpu().detach().numpy(), 0, -1)),
            wandb.Image(
                np.clip(
                    np.moveaxis(reconstructions[i].cpu().detach().numpy(),
                                0,
                                -1),
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
          os.path.join("checkpoints", "base_{}_{}.pth".format(args.beta, args.schedule))
        log_info = {
            "state_dict": model.state_dict(),
        }
        torch.save(log_info, save_checkpoint)

    wandb.finish()


if __name__ == "__main__":
    main()
