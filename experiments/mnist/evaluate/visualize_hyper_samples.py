import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli
import tqdm
import wandb

from experiments.init_wandb import init_wandb
from experiments.mnist.hyper_model_pipeline import build_hyper_criterion
from experiments.mnist.hyper_model_pipeline import build_hyper_model
from experiments.mnist.input_pipeline import build_input_queue
from src.config import HyperConfig
from src.config import TrainConfig
from src.evaluate import AverageMeter
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import tile_image
from src.evaluate import update_metric
from src.fid.fid_score import calculate_frechet_distance
from src.fid.fid_score import compute_statistics_of_generator
from src.fid.fid_score import load_statistics
from src.fid.inception import InceptionV3
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default="hypervae-mnist-hyper-train")

parser.add_argument("--encoder_name", type=str, default="cnn")
parser.add_argument("--decoder_name", type=str, default="cnn")

parser.add_argument("--block_type", type=str, default="mlp")
parser.add_argument("--preact_transform", type=int, default=0)
parser.add_argument("--preprocess_beta", type=int, default=1)
parser.add_argument("--include_sigmoid_activation", type=int, default=0)
parser.add_argument("--include_layer_norm", type=int, default=1)
parser.add_argument("--include_shift", type=int, default=1)
parser.add_argument("--include_residual_connection", type=int, default=1)
parser.add_argument("--sample_type", type=str, default="beta_log_uniform")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/")

args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def create_generator_vae(model, value, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            # logits = model.sample(batch_size, 1.0)
            # output = model.decoder_output(logits)
            output_img = model.prior_sample(
                value=value, batch_size=1, device=DEVICE)
            # output_img = output_img.mean()
            output_img = Bernoulli(logits=output_img).mean
            output_img = (output_img >= 0.5).float()
            # output_img = output_img.mean()
            # output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()


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
        model.load_state_dict(
            torch.load(f, map_location=torch.device("cpu"))["state_dict"])
    else:
        model.load_state_dict(torch.load(f)["state_dict"])

    print(model)
    model.eval()

    sample_lst = model.get_test_samples(20)
    for sample in sample_lst:
        n = 3
        m = 5
        num_samples = n * m

        output_img_lst = []
        for _ in range(num_samples):
            output_img = model.prior_sample(sample, DEVICE)
            output_img_lst.append(output_img)
        output_img = torch.concat(output_img_lst)
        # output_img = torch.sigmoid(output_img)
        output_img = Bernoulli(logits=output_img).mean
        output_img = (output_img >= 0.5).float()
        output_tiled = tile_image(output_img, n, m)
        plt.rcParams['figure.figsize'] = (12, 12)
        plt.imshow(
            output_tiled.detach().cpu().permute(1, 2, 0).numpy(), cmap="Greys")
        plt.title(sample)
        plt.show()
        # break

        print("hi")
        dims = 2048
        g = create_generator_vae(model, sample, 128, 50000)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        fid_dir = "/home/baejuhan/codes/hyper-vae/experiments/mnist/checkpoints"
        fid_model = InceptionV3([block_idx], model_dir=fid_dir).to(DEVICE)
        m, s = compute_statistics_of_generator(g, fid_model, 128, dims, DEVICE,
                                               max_samples=50000)
        path = os.path.join('../checkpoints/mnist.npz')
        m0, s0 = load_statistics(path)

        fid = calculate_frechet_distance(m0, s0, m, s)
        print("fid score")
        print(fid)


if __name__ == "__main__":
    main()
