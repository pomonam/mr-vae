import argparse
from math import pi
import os

from linear_utils import analytic_rate_and_distortion
from linear_vae import LinearDecoder
from linear_vae import LinearEncoder
from linear_vae import LinearHyperVae
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import tqdm
import wandb

from experiments.b_mnist.input_pipeline import build_input_queue
from experiments.b_mnist.model_pipeline import build_criterion
from experiments.b_mnist.results.rd_curve import get_rd
from experiments.init_wandb import init_api
from experiments.init_wandb import init_wandb
from src.config import HyperConfig
from src.config import TrainConfig
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric
from src.plotting import init_plotting
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default="hyper_vae-hyper-b_mnist_linear")

parser.add_argument(
    "--training_method",
    type=str,
    default="sequential",
    choices=["simultaneous", "sequential"])
parser.add_argument("--bottleneck_size", type=int, default=128)
parser.add_argument("--hyper_type", type=str, default="add")
parser.add_argument("--block_type", type=str, default="linear")
parser.add_argument("--include_output_layer", type=int, default=1)
parser.add_argument("--include_sigmoid_activation", type=int, default=1)
parser.add_argument("--preprocess_beta", type=int, default=0)
parser.add_argument("--sample_type", type=str, default="fixed_log_uniform")
parser.add_argument("--sample_range", type=tuple, default=(1e-3, 10))

parser.add_argument("--total_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch_size", type=int, default=128)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=500)

args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")

# ========
# Plotting
# ========

ENTITY = "lrscheduler114"
EXPERIMENT_NAME = "hyper_vae-hyper-b_mnist_linear"
ID = "k9m4y9yx"


def get_summary(summary):
    beta_to_rate = dict(
        zip(summary["train_eval/beta_lst"], summary["train_eval/rate_lst"]))
    beta_to_dist = dict(
        zip(summary["train_eval/beta_lst"], summary["train_eval/dist_lst"]))
    beta_to_elbo = dict(
        zip(summary["train_eval/beta_lst"], summary["train_eval/loss_lst"]))

    beta_to_rate_analytical = dict(
        zip(summary["analytical/beta_lst"], summary["analytical/rate_lst"]))
    beta_to_dist_analytical = dict(
        zip(summary["analytical/beta_lst"], summary["analytical/dist_lst"]))
    return beta_to_rate, beta_to_dist, beta_to_elbo, beta_to_rate_analytical, beta_to_dist_analytical


def plot_rd_curves():
    init_plotting()

    api = init_api()
    runs = api.runs(ENTITY + "/" + EXPERIMENT_NAME)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if run.state == "finished" and run.id == ID:
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith('_')})
            name_list.append(run.name)

    results = get_summary(summary_list[0])
    rate_dict, dist_dict, elbo_dict = results[0], results[1], results[2]
    analytical_rate_dict, analytical_dist_dict = results[3], results[4]

    keys = rate_dict.keys()
    values = zip(rate_dict.values(), dist_dict.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    dist = np.array([c[1] for c in combined_dict.values()])
    plt.plot(rate, dist, label="Hypernetwork")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")
    plt.scatter(rate, dist)

    keys = analytical_rate_dict.keys()
    values = zip(analytical_rate_dict.values(), analytical_dist_dict.values())
    analytical_combined_dict = dict(zip(keys, values))

    analytical_rate = np.array(
        [c[0] for c in analytical_combined_dict.values()])
    analytical_dist = np.array(
        [c[1] for c in analytical_combined_dict.values()])
    plt.plot(analytical_rate, analytical_dist, label='Analytical')

    min_val = min(
        np.min(rate),
        np.min(dist),
        np.min(analytical_rate),
        np.min(analytical_dist)) - 10
    max_val = min(
        np.max(rate),
        np.max(dist),
        np.max(analytical_rate),
        np.max(analytical_dist)) + 10
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Rate")
    plt.ylabel("Distortion")

    rate, dist = get_rd("hv-b_mnist_mlp_train-v5")
    plt.plot(rate, dist, label="Baseline")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")

    plt.legend()
    plt.show()


# =====================
# Hypernetwork Training
# =====================


def hyper_evaluate(model, criterion, epoch, name):
    model.eval()

    with torch.no_grad():
        beta_lst = np.logspace(-3, 1, num=20)
        loss_lst = []
        rate_lst = []
        dist_lst = []

        if name == "analytical":
            for beta in tqdm.tqdm(beta_lst):
                loader = build_input_queue(name, args.batch_size, DEVICE)
                rate, dist = analytic_rate_and_distortion(model, loader, beta)
                rate_lst.append(rate)
                dist_lst.append(dist)
                loss_lst.append(rate + dist)
        else:
            for beta in beta_lst:
                metric_dict = initialize_metric(criterion.get_metric_lst())
                loader = build_input_queue(name, args.batch_size, DEVICE)
                p_bar = tqdm.tqdm(loader)

                for batch in p_bar:
                    inputs = batch["inputs"]
                    output_dict = model.fixed_forward(inputs, beta)
                    # We want to compute exact ELBO here
                    _, loss_dict = criterion.eval_forward(output_dict)

                    metric_dict = update_metric(metric_dict,
                                                loss_dict,
                                                inputs.size(0))
                    summ_dict = summarize_metric(metric_dict)
                    summ_str = generate_metric_str(name, epoch, summ_dict)
                    p_bar.set_description(summ_str)

                summ_dict = summarize_metric(metric_dict, name="")
                loss_lst.append(summ_dict["loss"])
                rate_lst.append(summ_dict["rate"])
                dist_lst.append(summ_dict["distortion"])

        wandb.log({
            f"{name}/loss_lst": loss_lst,
            f"{name}/rate_lst": rate_lst,
            f"{name}/dist_lst": dist_lst,
            f"{name}/beta_lst": beta_lst
        })

        rd_data = [[x, y] for (x, y) in zip(rate_lst, dist_lst)]
        table = wandb.Table(data=rd_data, columns=["rate", "distortion"])
        wandb.log({
            f"{name}/rd_curve":
                wandb.plot.line(table, "rate", "distortion", title="RD Curve")
        })

        loss_lst = np.array(loss_lst)
        rate_lst = np.array(rate_lst)
        dist_lst = np.array(dist_lst)

        total_auc = np.sum(loss_lst)
        top_auc = np.sum(loss_lst[:5])
        bot_auc = np.sum(loss_lst[5:])
        auc_dict = {
            f"{name}/total_auc": total_auc,
            f"{name}/top_auc": top_auc,
            f"{name}/bot_auc": bot_auc,
            f"{name}/max_rate": np.max(rate_lst),
            f"{name}/min_rate": np.min(rate_lst),
            f"{name}/abs_rate": np.max(rate_lst) - np.min(rate_lst),
            f"{name}/max_dist": np.max(dist_lst),
            f"{name}/min_dist": np.min(dist_lst),
            f"{name}/abs_dist": np.max(dist_lst) - np.min(dist_lst),
        }
        wandb.log(auc_dict)


def hyper_train(model, biq, criterion, optimizer, cfg, hyper_cfg):
    do_checkpoint = cfg.checkpoint_dir is not None
    if do_checkpoint and os.path.exists(
            os.path.join(args.checkpoint_dir, "checkpoint.pth")):
        slurm_checkpoint = torch.load(
            os.path.join(args.checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(slurm_checkpoint["state_dict"])
        optimizer.load_state_dict(slurm_checkpoint["optimizer"])
        epoch = slurm_checkpoint["epoch"]
    else:
        epoch = 0

    while epoch < cfg.total_epochs:
        do_evaluate = epoch % cfg.eval_freq == 0 and epoch != 0
        do_save = epoch % cfg.save_freq == 0 and epoch != 0

        if do_evaluate:
            hyper_evaluate(model, criterion, epoch, "train_eval")
            hyper_evaluate(model, criterion, epoch, "test")

        if do_checkpoint and do_save:
            slurm_check_dir = os.path.join(args.checkpoint_dir,
                                           "checkpoint.pth")
            log_info = {
                "id": wandb.run.id,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(log_info, slurm_check_dir)

        model.train()
        loader = biq("train", cfg.batch_size, DEVICE)
        p_bar = tqdm.tqdm(loader)
        metric_dict = initialize_metric(criterion.get_metric_lst())

        for batch in p_bar:
            inputs = batch["inputs"]
            output_dict = model.sample_forward(inputs)
            loss, loss_dict = criterion(output_dict, output_dict["beta"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
            summ_dict = summarize_metric(metric_dict)
            summ_str = generate_metric_str("train", epoch, summ_dict)
            p_bar.set_description(summ_str)

        summ_dict = summarize_metric(metric_dict, name="train_step/")
        wandb.log(summ_dict)
        epoch = epoch + 1

        if np.isnan(summ_dict["train_step/loss"]):
            wandb.finish(exit_code=1)
            raise ValueError()


def main():
    init_wandb(
        args.checkpoint_dir,
        project_name=args.experiment_name,
        config=vars(args))
    cfg = TrainConfig(args)
    hyper_cfg = HyperConfig(args)

    seed_everything(cfg.seed)
    encoder = LinearEncoder(bottleneck_size=args.bottleneck_size).to(DEVICE)
    decoder = LinearDecoder(bottleneck_size=args.bottleneck_size).to(DEVICE)
    model = LinearHyperVae(encoder, decoder, hyper_cfg).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = build_criterion(DEVICE)
    hyper_train(model, build_input_queue, criterion, optimizer, cfg, hyper_cfg)

    hyper_evaluate(model, criterion, cfg.total_epochs, "analytical")
    hyper_evaluate(model, criterion, cfg.total_epochs, "train_eval")
    hyper_evaluate(model, criterion, cfg.total_epochs, "test")


if __name__ == "__main__":
    main()
    #plot_rd_curves()