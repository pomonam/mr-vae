import argparse
import os

import numpy as np
import torch
import tqdm
import wandb
import matplotlib.pyplot as plt

os.environ["WANDB_API_KEY"] = "65a71cb86f66a117460fb632080693d4cc9ab979"

from input_pipeline import build_input_queue
from experiments.wandb_utils import init_wandb
from src.config import TrainConfig
from src.evaluate import initialize_metric
from src.evaluate import generate_metric_str
from src.evaluate import summarize_metric
from src.evaluate import update_metric
from src.utils import seed_everything

from linear_utils import analytic_rate_and_distortion
from linear_beta_vae import LinearVae
from linear_beta_vae import LinearDecoder
from linear_beta_vae import LinearEncoder

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default="linear_beta_vae-mnist_baseline")

parser.add_argument("--bottleneck_size", type=int, default=100)
parser.add_argument("--total_epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=5000) # Also corresponds to dataset size
parser.add_argument("--beta", type=float, required=True)
parser.add_argument("--schedule", type=str, default="constant")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=50)
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def evaluate(model, biq, epoch, name, delta=0.01):
    model.eval()

    with torch.no_grad():
        loader = biq(name, args.batch_size, DEVICE)
        p_bar = tqdm.tqdm(loader)
        metric_dict = initialize_metric(["loss", "rate", "distortion"])
        means = []

        if name == "analytical":
            rate, dist = analytic_rate_and_distortion(model, loader, 1/args.beta)
            wandb.log({name + "/" + "rate": rate, name + "/" + "dist": dist})
        else:
            for batch in p_bar:
                inputs = batch["inputs"]
                elbo, output_dict, loss_dict = model(inputs)
                means.append(output_dict["mean"])

                metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
                summ_dict = summarize_metric(metric_dict)
                summ_str = generate_metric_str(name, epoch, summ_dict)
                p_bar.set_description(summ_str)

            means = torch.cat(means, dim=0)

            summ_dict = summarize_metric(metric_dict, name=name + "/")
            wandb.log(summ_dict)


def train(model, biq, optimizer, scheduler, cfg):
    do_checkpoint = cfg.checkpoint_dir is not None
    if do_checkpoint and os.path.exists(
            os.path.join(cfg.checkpoint_dir, "checkpoint.pth")):
        slurm_checkpoint = torch.load(
            os.path.join(cfg.checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(slurm_checkpoint["state_dict"])
        optimizer.load_state_dict(slurm_checkpoint["optimizer"])
        epoch = slurm_checkpoint["epoch"]
    else:
        epoch = 0

    while epoch < cfg.total_epochs:
        do_evaluate = epoch % cfg.eval_freq == 0
        do_save = epoch % cfg.save_freq == 0 and epoch != 0

        if do_evaluate:
            evaluate(model, biq, epoch, "train_eval")
            evaluate(model, biq, epoch, "test")

        if do_checkpoint and do_save:
            slurm_check_dir = os.path.join(cfg.checkpoint_dir, "checkpoint.pth")
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
        metric_dict = initialize_metric(["loss", "rate", "distortion"])

        for batch in p_bar:
            optimizer.zero_grad()
            inputs = batch["inputs"]
            elbo, output_dict, loss_dict = model(inputs, beta=cfg.get_beta(epoch))
            loss = -elbo
            loss.backward()
            optimizer.step()

            metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
            summ_dict = summarize_metric(metric_dict)
            summ_str = generate_metric_str("train", epoch, summ_dict)
            p_bar.set_description(summ_str)

        scheduler.step()
        summ_dict = summarize_metric(metric_dict, name="train_step/")
        summ_dict["beta"] = cfg.get_beta(epoch)
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

    seed_everything(args.seed)
    cfg = TrainConfig(args)

    # Create Model
    encoder = LinearEncoder(bottleneck_size=args.bottleneck_size).to(DEVICE)
    decoder = LinearDecoder(bottleneck_size=args.bottleneck_size).to(DEVICE)
    model = LinearVae(encoder, decoder).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    train(model, build_input_queue, optimizer, lr_scheduler, cfg)
    evaluate(model, build_input_queue, cfg.total_epochs, "train_eval")
    evaluate(model, build_input_queue, cfg.total_epochs, "test")
    evaluate(model, build_input_queue, cfg.total_epochs, "analytical")

    '''
    # Visualizing the reconstruction
    test_loader = build_input_queue("test", cfg.batch_size, DEVICE)
    test_batch = next(test_loader)
    outputs_dict = model.forward(test_batch["inputs"])
    logits = outputs_dict["logits"].view(-1, 28, 28)
    plt.figure(figsize=(5, 5))
    plt.axis("square")
    for i in range(50):
        data_i = test_batch["inputs"].view(-1, 28, 28)[i].data.cpu().numpy()
        recon_i = torch.sigmoid(logits[i]).data.cpu().numpy()
        plt.subplot(10, 10, 2 * i + 1)
        plt.imshow(data_i, cmap="Greys")
        plt.axis("off")
        plt.subplot(10, 10, 2 * i + 2)
        plt.imshow(recon_i, cmap="Greys")
        plt.axis("off")
    wandb.log({"reconstruction": plt})
    '''
    wandb.finish()


if __name__ == "__main__":
    main()
