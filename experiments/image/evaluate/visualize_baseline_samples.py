import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli
from experiments.image.input_pipeline import load_data


from experiments.image.model_pipeline import build_model
from src.config import TrainConfig

from src.evaluate import tile_image
from src.fid.fid_score import calculate_frechet_distance
from src.fid.fid_score import compute_statistics_of_generator
from src.fid.fid_score import load_statistics
from src.fid.inception import InceptionV3
from src.utils import seed_everything
import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb


parser = argparse.ArgumentParser()


parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints/")
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            output_img = model.prior_sample(batch_size=batch_size, device=DEVICE)
            output_img = Bernoulli(logits=output_img).mean
            output_img = (output_img >= 0.5).float()
        yield output_img.float()


def main():
    cfg = TrainConfig(args)
    file_lst = glob.glob(os.path.join(args.checkpoint_dir, "base_*"))
    batch_size = 128
    num_column = 5

    prior_dict = {}
    reconst_dict = {}
    fid_dict = {}

    seed_everything(cfg.seed)
    loader = load_data(
        "cifar", "test", num_column, data_path="../../../logs/data")
    sample_inputs = iter(loader).next()
    sample_inputs = sample_inputs[0]

    for f in file_lst:
        if "cyclic" in f:
            model = build_model("cifar", DEVICE)

            if DEVICE == torch.device("cpu"):
                model.load_state_dict(
                    torch.load(f,
                               map_location=torch.device("cpu"))["state_dict"])
            else:
                model.load_state_dict(torch.load(f)["state_dict"])

            model.eval()

            # Get fixed latent ...
            outputs_dict = {
                "mean": torch.zeros((num_column, 64)).to(DEVICE),
                "log_var": torch.zeros((num_column, 64)).to(DEVICE),
            }
            seed_everything(cfg.seed)
            prior_z = model.sampler.sample(outputs_dict)

            # Get reconstruction latent ...
            outputs_dict = model.encode(sample_inputs.reshape(num_column, 3, 32, 32).to(DEVICE))
            reconst_z = model.sampler.sample(outputs_dict)

            beta = float(f.split("_")[1])
            prior_img = model.decoder(prior_z)
            prior_img = torch.clamp(prior_img, -1., 1.)
            prior_img = prior_img / 2. + 0.5
            prior_dict[beta] = prior_img

            reconst_img = model.decoder(reconst_z)
            reconst_img = torch.clamp(reconst_img, -1., 1.)
            reconst_img = reconst_img / 2. + 0.5
            reconst_dict[beta] = reconst_img

            # dims = 2048
            # g = create_generator_vae(model, batch_size, 50000)
            # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            # fid_dir = "/home/baejuhan/codes/hyper-vae/experiments/mnist/checkpoints"
            # model = InceptionV3([block_idx], model_dir=fid_dir).to(DEVICE)
            # m, s = compute_statistics_of_generator(g, model, batch_size, dims, DEVICE,
            #                                        max_samples=50000)
            # path = os.path.join('../checkpoints/mnist.npz')
            # m0, s0 = load_statistics(path)
            # fid = calculate_frechet_distance(m0, s0, m, s)
            # fid_dict[beta] = fid

    plt.rcParams.update({"figure.dpi": 150})
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
    plt.title(r"Samples Generated for Different $\beta$")
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
    plt.title(r"Reconstruction for Different $\beta$")
    plt.show()

    # sorted_fid_dict = dict(
    #     sorted(fid_dict.items(), key=lambda item: item[0]))
    # print("FID:")
    # print(list(sorted_fid_dict.values()))


if __name__ == "__main__":
    main()
