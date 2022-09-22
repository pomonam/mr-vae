import argparse
import copy
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
from experiments.binary_image.baseline_train import build_model
import matplotlib.pyplot as plt
from experiments.binary_image.hyper_train import build_model as build_hyper_model
from src.config import HyperConfig

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def main():
  beta = 1.0
  model = build_model("resnet", "resnet", DEVICE)
  print(model)
  save_checkpoint = \
    os.path.join("../checkpoints", "final_base_{}_{}_{}_{}_{}.pth".
                 format("omniglot", "resnet",
                        "resnet", str(beta), "monotonic"))
  checkpoint = torch.load(save_checkpoint)
  model.load_state_dict(checkpoint["state_dict"])
  base_model = copy.deepcopy(model)

  dist_lst = []
  betas = np.logspace(-2, 1, num=10, base=10)
  for beta in betas:
    model = build_model("resnet", "resnet", DEVICE)
    save_checkpoint = \
      os.path.join("../checkpoints", "final_base_{}_{}_{}_{}_{}.pth".
                   format("omniglot", "resnet",
                          "resnet", str(beta), "monotonic"))
    checkpoint = torch.load(save_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    base_params = list(model.parameters())
    hyper_params = list(base_model.parameters())
    dist = 0
    for i in range(len(base_params)):
      dist += torch.sum((base_params[i] - hyper_params[i]) ** 2.)
    dist_lst.append(dist.item())
  print(dist_lst)
  plt.plot(betas, dist_lst)
  plt.show()

  cfg = HyperConfig(None)
  cfg.initialize_default_config()
  model = build_hyper_model("resnet", "resnet", cfg, DEVICE)
  print(model)
  save_checkpoint = \
    os.path.join("../checkpoints", "hyper_{}_{}_{}.pth".
                 format("omniglot", "resnet",
                        "resnet"))
  checkpoint = torch.load(save_checkpoint)
  model.load_state_dict(checkpoint["state_dict"])
  base_model = copy.deepcopy(model)

if __name__ == "__main__":
  main()
