import argparse
import os

import numpy as np
import torch
import tqdm
import wandb
import glob
from src.evaluate import tile_image
from src.utils import seed_everything
import matplotlib.pyplot as plt
from experiments.mnist.input_pipeline import build_input_queue
from experiments.mnist.hyper_model_pipeline import build_hyper_criterion
from experiments.mnist.hyper_model_pipeline import build_hyper_model
from experiments.init_wandb import init_wandb
from src.config import HyperConfig
from src.config import TrainConfig
from src.evaluate import AverageMeter

from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="hypervae-mnist-hyper-train")

parser.add_argument("--encoder_name", type=str, default="cnn")
parser.add_argument("--decoder_name", type=str, default="cnn")

parser.add_argument("--block_type", type=str, default="mlp")
parser.add_argument("--preact_transform", type=int, default=1)
parser.add_argument("--include_sigmoid_activation", type=int, default=0)
parser.add_argument("--include_layer_norm", type=int, default=0)
parser.add_argument("--include_shift", type=int, default=1)
parser.add_argument("--include_residual_connection", type=int, default=1)
parser.add_argument("--preprocess_beta", type=int, default=1)
parser.add_argument("--sample_type", type=str, default="beta_log_uniform")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/")

args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")



def main():

    cfg = TrainConfig(args)
    hyper_cfg = HyperConfig(args)

    file_lst = glob.glob(os.path.join(args.checkpoint_dir, "hyper*"))

    seed_everything(cfg.seed)
    model = build_hyper_model(args.encoder_name,
                              args.decoder_name,
                              hyper_cfg,
                              DEVICE)

    print(file_lst)
    f = file_lst[0]
    print(f)
    if DEVICE == torch.device("cpu"):
        model.load_state_dict(torch.load(f, map_location=torch.device("cpu"))["state_dict"])
    else:
        model.load_state_dict(torch.load(f)["state_dict"])

    print(model)
    model.eval()
    n = 3
    m = 5
    num_samples = n * m

    sample_lst = model.get_test_samples(20)

    for sample in sample_lst:
        output_img_lst = []
        for _ in range(num_samples):
            output_img = model.prior_sample(sample)
            output_img_lst.append(output_img)
        output_img = torch.concat(output_img_lst)
        output_img = torch.sigmoid(output_img)
        output_tiled = tile_image(output_img, n, m)
        plt.rcParams['figure.figsize'] = (12, 12)
        plt.imshow(output_tiled.detach().cpu().permute(1, 2, 0).numpy(), cmap="Greys")
        plt.title(sample)
        plt.show()
        # break


if __name__ == "__main__":
    main()
