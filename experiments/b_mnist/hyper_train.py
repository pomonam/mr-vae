import argparse
import os

import numpy as np
import torch
import tqdm
import wandb

from experiments.b_mnist.input_pipeline import build_input_queue
from experiments.b_mnist.model_pipeline import build_criterion
from experiments.b_mnist.model_pipeline import build_hyper_model
from experiments.evaluate import generate_metric_str
from experiments.evaluate import initialize_metric
from experiments.evaluate import summarize_metric
from experiments.evaluate import update_metric
from experiments.init_wandb import init_wandb
from experiments.utils import seed_everything
from src.config import HyperConfig

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="hyper_vae-b_mnist_mlp")

parser.add_argument("--encoder_name", type=str, default="resnet")
parser.add_argument("--decoder_name", type=str, default="cnn")

# Configurations specific to the hypernet ...
parser.add_argument("--training_method", type=str, default="simultaneous",
                    choices=["simultaneous", "sequential"])
parser.add_argument("--hyper_type", type=str, default="add")
parser.add_argument("--block_type", type=str, default="linear")
parser.add_argument("--include_output_linear", type=int, default=1)
parser.add_argument("--include_sigmoid_activation", type=int, default=1)
parser.add_argument("--sample_type", type=str, default="uniform")
parser.add_argument("--sample_range", type=tuple, default=(0, 10))

parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--schedule", type=str, default="constant")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=100)
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def hyper_evaluate(model, criterion, epoch, name):
    model.eval()

    with torch.no_grad():
        beta_lst = np.logspace(-5, 1, num=20)
        loss_lst = []
        rate_lst = []
        dist_lst = []

        for beta in beta_lst:
            loader = build_input_queue(name, args.batch_size, DEVICE)
            p_bar = tqdm.tqdm(loader)
            metric_dict = initialize_metric(criterion.get_metric_lst())

            for batch in p_bar:
                inputs = batch["inputs"]
                output_dict = model.fixed_forward(inputs, beta)
                ones_beta = model.fixed_beta(inputs, 1)
                # We want to compute exact ELBO here
                _, loss_dict = criterion(output_dict, beta=ones_beta)

                metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
                summ_dict = summarize_metric(metric_dict)
                summ_str = generate_metric_str(name, epoch, summ_dict)
                p_bar.set_description(summ_str)

            summ_dict = summarize_metric(metric_dict, name="")
            loss_lst.append(summ_dict["loss"])
            rate_lst.append(summ_dict["rate"])
            dist_lst.append(summ_dict["distortion"])

        rd_data = [[x, y] for (x, y) in zip(rate_lst, dist_lst)]
        table = wandb.Table(data=rd_data, columns=["rate", "distortion"])
        wandb.log({f"{name}/rd_curve":
                   wandb.plot.line(table, "rate", "distortion", title="RD Curve")})

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


def hyper_train(model, optimizer, criterion):
    if args.checkpoint_dir is not None and os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint.pth")):
        slurm_checkpoint = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(slurm_checkpoint["state_dict"])
        optimizer.load_state_dict(slurm_checkpoint["optimizer"])
        epoch = slurm_checkpoint["epoch"]
    else:
        epoch = 0

    while epoch < args.epochs:
        if epoch % args.eval_freq == 0 and epoch != 0:
            hyper_evaluate(model, criterion, epoch, "train_eval")
            hyper_evaluate(model, criterion, epoch, "test")

        if args.checkpoint_dir is not None:
            slurm_check_dir = os.path.join(args.checkpoint_dir, "checkpoint.pth")
            log_info = {
                "id": wandb.run.id,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(log_info, slurm_check_dir)

        model.train()
        loader = build_input_queue("train", args.batch_size, DEVICE)
        p_bar = tqdm.tqdm(loader)
        metric_dict = initialize_metric(criterion.get_metric_lst())

        for batch in p_bar:
            inputs = batch["inputs"]

            if args.training_method == "sequential":
                output_dict = model.hyper_ignore_forward(inputs)
                loss, loss_dict = criterion(output_dict, output_dict["beta"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
    init_wandb(args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))

    hyper_config = HyperConfig(args)
    seed_everything(args.seed)
    model = build_hyper_model(args.encoder_name, args.decoder_name, hyper_config, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = build_criterion(DEVICE)
    hyper_train(model, optimizer, criterion)
    hyper_evaluate(model, criterion, args.epochs, "train_eval")
    hyper_evaluate(model, criterion, args.epochs, "test")


if __name__ == "__main__":
    main()
