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
from experiments.omniglot.input_pipeline import load_data
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb


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
            output_img = model.prior_sample(
                value=value, batch_size=batch_size, device=DEVICE)
            output_img = Bernoulli(logits=output_img).mean
            output_img = (output_img >= 0.5).float()
        yield output_img.float()


def main():
    cfg = TrainConfig(args)
    hyper_cfg = HyperConfig(args)
    file_lst = glob.glob(os.path.join(args.checkpoint_dir, "hyper*"))
    batch_size = 128
    num_column = 5

    prior_dict = {}
    reconst_dict = {}
    fid_dict = {}

    seed_everything(cfg.seed)
    loader = load_data(
        "test", num_column, data_path="../../../logs/", force_shuffle=True)
    sample_inputs = iter(loader).next()
    sample_inputs = sample_inputs[0]

    seed_everything(cfg.seed)
    model = build_hyper_model(args.encoder_name,
                              args.decoder_name,
                              hyper_cfg,
                              DEVICE)

    f = file_lst[0]

    if DEVICE == torch.device("cpu"):
        model.load_state_dict(
            torch.load(f, map_location=torch.device("cpu"))["state_dict"])
    else:
        model.load_state_dict(torch.load(f)["state_dict"])

    model.eval()

    sample_lst = model.get_test_samples(20)
    for sample in sample_lst:
        sample_dict = model.sample_inverse(torch.Tensor([0]).to(DEVICE), sample)
        model.set_net_inputs(sample_dict["net"])

        # Get fixed latent ...
        outputs_dict = {
            "mean": torch.zeros((num_column, 32)).to(DEVICE),
            "log_var": torch.zeros((num_column, 32)).to(DEVICE),
        }
        seed_everything(cfg.seed)
        prior_z = model.sampler.sample(outputs_dict)

        # Get reconstruction latent ...
        outputs_dict = model.encode(sample_inputs.reshape(num_column, 1, 28, 28).to(DEVICE))
        reconst_z = model.sampler.sample(outputs_dict)

        prior_img = model.decoder(prior_z)
        prior_img = Bernoulli(logits=prior_img).mean
        prior_img = (prior_img >= 0.5).float()
        prior_dict[sample] = prior_img

        reconst_img = model.decoder(reconst_z)
        reconst_img = Bernoulli(logits=reconst_img).mean
        reconst_img = (reconst_img >= 0.5).float()
        reconst_dict[sample] = reconst_img

        dims = 2048
        g = create_generator_vae(model, sample, 128, 50000)
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        fid_dir = "/home/baejuhan/codes/hyper-vae/experiments/omniglot/checkpoints"
        fid_model = InceptionV3([block_idx], model_dir=fid_dir).to(DEVICE)
        m, s = compute_statistics_of_generator(g, fid_model, 128, dims, DEVICE,
                                               max_samples=50000)
        path = os.path.join('../checkpoints/omniglot.npz')
        m0, s0 = load_statistics(path)
        fid = calculate_frechet_distance(m0, s0, m, s)
        fid_dict[sample] = fid

    plt.rcParams.update({"figure.dpi": 300})
    plt.rcParams.update(bundles.aistats2022(column="full"))
    plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
    plt.rcParams.update(markers.inverted())

    @plt.FuncFormatter
    def fake_log(x, pos):
        y = x / 500
        y = float(y)
        return r'$10^{}$'.format(round(y, 2))

    sorted_prior_dict = dict(
        sorted(prior_dict.items(), key=lambda item: item[0]))
    prior_imgs = torch.concat(list(sorted_prior_dict.values()))
    output_tiled = tile_image(prior_imgs, 20, num_column)
    plt.imshow(
        output_tiled.detach().cpu().permute(2, 1, 0).numpy(),
        cmap="Greys",
    )
    plt.yticks([])
    plt.tight_layout()
    plt.title(r"Samples Generated by Hypernetwork for Different $\beta$")
    plt.show()

    fig, ax = plt.subplots()
    sorted_reconst_dict = dict(
        sorted(reconst_dict.items(), key=lambda item: item[0]))
    reconst_imgs = torch.concat(list(sorted_reconst_dict.values()))
    output_tiled = tile_image(reconst_imgs, 20, num_column)
    plt.imshow(
        output_tiled.detach().cpu().permute(2, 1, 0).numpy(),
        cmap="Greys",
    )
    plt.yticks([])
    plt.tight_layout()
    plt.title(r"Reconstruction Generated by Hypernetwork for Different $\beta$")
    plt.show()

    sorted_fid_dict = dict(
        sorted(fid_dict.items(), key=lambda item: item[0]))
    # BASELINE_FID = [411.5334972779568, 374.7136443060085, 336.3524864339188, 307.89274905654565, 261.23468987194724, 208.8394285527536, 169.69655640720927, 148.3969630809807, 122.55783175298882, 83.55142391986055, 59.15827241597475, 40.666423185858804, 26.517076562018133, 23.145955818063328, 23.727066888601087, 26.879348419054594, 29.692440821263403, 32.840371276666815, 37.35687236530222, 38.41171559414312]
    BASELINE_FID = [416.0140316795957, 403.46994942198216, 390.32596769026355, 350.91594817179276, 315.5376687802361, 277.4233529858552, 226.41428828618254, 171.5560408947034, 134.72618538711635, 106.01848493836134, 81.43947880159504, 70.94082528274828, 66.80339327430207, 64.71352382711552, 61.87727040325183, 60.05144043491839, 61.81621839828347, 115.69763436154868, 377.56221667395255, 422.92757420271937]

    plt.plot([0], [0])
    plt.scatter(
        sorted_fid_dict.keys(),
        BASELINE_FID,
        label=r"Independent Training",
        edgecolors="k",
        linewidths=0.5,
        c=rgb.tue_lightblue)
    plt.plot(
        sorted_fid_dict.keys(), sorted_fid_dict.values(), "o-", label="Hypernetwork",
        linewidth=2, c=rgb.tue_ocre)

    plt.xlabel(r"$\beta$")
    plt.xscale("log")
    plt.ylabel("FID")

    plt.title(r"$\beta$ vs. FID for Omniglot")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
