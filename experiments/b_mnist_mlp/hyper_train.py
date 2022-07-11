import argparse
import os

import torch
from torch.backends import cudnn
import tqdm
import wandb
import numpy as np
from experiments.b_mnist_mlp.input_pipeline import build_input_queue
from experiments.b_mnist_mlp.model_pipeline import build_criterion
from experiments.b_mnist_mlp.model_pipeline import build_hyper_model
from experiments.evaluate import generate_metric_str
from experiments.evaluate import initialize_metric
from experiments.evaluate import summarize_metric
from experiments.evaluate import update_metric
from experiments.init_wandb import init_wandb
from experiments.utils import seed_everything
from src.utils import sample_beta

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="hyper_vae-b_mnist_mlp")

parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--training_method", type=str, default="sequential",
                    choices=["simultaneous", "sequential"])

parser.add_argument("--hyper_type", type=str, default="add")
parser.add_argument("--param_method", type=str, default="mlp")
parser.add_argument("--apply_exp", type=int, default=1)

parser.add_argument("--no_cuda", type=bool, default=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=10)
args = parser.parse_args()

cuda = torch.cuda.is_available() and not args.no_cuda
DEVICE = torch.device("cuda" if cuda else "cpu")
cudnn.benchmark = True


def hyper_evaluate(model, criterion, epoch, name):
    model.eval()

    with torch.no_grad():
        beta_lst = np.linspace(0.001, 10, num=20)
        for beta in beta_lst:
            loader = build_input_queue(name, args.batch_size, DEVICE)
            p_bar = tqdm.tqdm(loader)
            metric_dict = initialize_metric(criterion.get_metric_lst())

            for batch in p_bar:
                inputs = batch["inputs"]
                betas = sample_beta(inputs.shape, sample_range=(beta, beta), device=DEVICE,
                                    apply_exp=args.apply_exp)
                output_dict = model(inputs, beta=betas)
                ones_betas = torch.ones((inputs.shape[0], 1)).to(DEVICE)
                _, loss_dict = criterion(output_dict, beta=ones_betas)

                metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
                summ_dict = summarize_metric(metric_dict)
                summ_str = generate_metric_str(name, epoch, summ_dict)
                p_bar.set_description(summ_str)

            summ_dict = summarize_metric(metric_dict, name=name + "/{}/".format(beta))
            wandb.log(summ_dict)


def hyper_train(model, optimizer, criterion):
    if args.checkpoint_dir is not None and os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint.pth")):
        slurm_checkpoint = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(slurm_checkpoint["state_dict"])
        optimizer.load_state_dict(slurm_checkpoint["optimizer"])
        epoch = slurm_checkpoint["epoch"]
    else:
        epoch = 0

    while epoch < args.epochs:
        if epoch % args.eval_freq == 0:
            hyper_evaluate(model, criterion, epoch, "train_eval")
            hyper_evaluate(model, criterion, epoch, "test")

        if args.checkpoint_dir is not None and epoch % args.save_freq == 0 and epoch != 0:
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
                betas = torch.zeros(inputs.shape[0], 1).to(DEVICE)
                output_dict = model(inputs, betas, ignore_hyper=True)
                loss, loss_dict = criterion(output_dict, betas)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            betas = sample_beta(inputs.shape, sample_range=(0.001, 10), device=DEVICE,
                                apply_exp=args.apply_exp)
            output_dict = model(inputs, betas)
            loss, loss_dict = criterion(output_dict, torch.exp(betas) if args.apply_exp else betas)
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

    seed_everything(args.seed)
    model = build_hyper_model(args.hyper_type, args.param_method, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = build_criterion(DEVICE)
    hyper_train(model, optimizer, criterion)
    hyper_evaluate(model, criterion, args.epochs, "train_eval")
    hyper_evaluate(model, criterion, args.epochs, "test")

    wandb.finish()


if __name__ == "__main__":
    main()
