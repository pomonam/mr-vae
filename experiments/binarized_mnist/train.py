import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import tqdm
import wandb

from experiments.init_wandb import init_wandb
from experiments.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="mlp_mnist_train")

parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--optimizer", type=str, default="sgdm")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--wd", type=float, default=0.)

parser.add_argument("--no_cuda", type=bool, default=False)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=10)
args = parser.parse_args()

cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device("cuda" if cuda else "cpu")
cudnn.benchmark = True


def train():
    pass


def main():
    init_wandb(args.checkpoint_dir,
               project_name=args.experiment_name,
               config=vars(args))
    seed_everything(args.seed)

    # Load model

    # Load optimizer
    optimizer = None

    train()
