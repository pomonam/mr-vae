import argparse
import functools
import os

import numpy as np
import torch
import tqdm
import wandb
from src.config import HyperConfig

from src.evaluate import AverageMeter
from experiments.image.input_pipeline import build_input_queue
from experiments.image.hyper_model_pipeline import build_hyper_criterion
from experiments.image.hyper_model_pipeline import build_hyper_model
from experiments.init_wandb import init_wandb
from src.config import TrainConfig
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="hypervae-image-train")

parser.add_argument("--data_name", type=str, default="cifar")

parser.add_argument("--block_type", type=str, default="mlp")
parser.add_argument("--preact_transform", type=int, default=0)
parser.add_argument("--include_sigmoid_activation", type=int, default=0)
parser.add_argument("--include_layer_norm", type=int, default=1)
parser.add_argument("--include_output_layer", type=int, default=1)
parser.add_argument("--include_shift", type=int, default=1)
parser.add_argument("--include_residual_connection", type=int, default=1)
# parser.add_argument("--include_chunk", type=int, default=0)
parser.add_argument("--preprocess_beta", type=int, default=0)
parser.add_argument("--sample_type", type=str, default="beta_log_uniform")

parser.add_argument("--total_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--beta", type=float, default=1)
parser.add_argument("--schedule", type=str, default="constant")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_freq", type=int, default=100)
parser.add_argument("--eval_freq", type=int, default=50)
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


def hyper_evaluate(model, criterion, epoch, name, delta=0.01):
    model.eval()

    with torch.no_grad():
        sample_lst = model.get_test_samples(20)
        loss_lst = []
        rate_lst = []
        dist_lst = []

        for sample in sample_lst:
            loader = build_input_queue(args.data_name, name, args.batch_size, DEVICE)
            p_bar = tqdm.tqdm(loader)
            mi_meter = AverageMeter()
            metric_dict = initialize_metric(criterion.get_metric_lst())
            means = []

            for batch in p_bar:
                inputs = batch["inputs"]
                output_dict = model.inverse_forward(inputs, sample)
                means.append(output_dict["mean"])
                mutual_info = model.calc_mi(inputs)
                mi_meter.update(mutual_info, inputs.size(0))

                # We want to compute exact ELBO here
                _, loss_dict = criterion.eval_forward(output_dict)
                metric_dict = update_metric(metric_dict,
                                            loss_dict,
                                            inputs.size(0))
                summ_dict = summarize_metric(metric_dict)
                summ_str = generate_metric_str(name, epoch, summ_dict)
                p_bar.set_description(summ_str)

            summ_dict = summarize_metric(metric_dict, name="")

            means = torch.cat(means, dim=0)
            au_mean = means.mean(0, keepdim=True)

            au_var = means - au_mean
            ns = au_var.size(0)
            au_var = (au_var**2).sum(dim=0) / (ns - 1)

            loss_lst.append(summ_dict["loss"])
            rate_lst.append(summ_dict["rate"])
            dist_lst.append(summ_dict["distortion"])

            wandb.log({
                f"{name}/sample": sample,
                f"{name}/au": (au_var >= delta).sum().item(),
                f"{name}/mi": mi_meter.avg
            })

        wandb.log({
            f"{name}/loss_lst": loss_lst,
            f"{name}/rate_lst": rate_lst,
            f"{name}/dist_lst": dist_lst,
            f"{name}/sample_lst": sample_lst,
        })

        rd_data = [[x, y] for (x, y) in zip(rate_lst, dist_lst)]
        table = wandb.Table(data=rd_data, columns=["rate", "distortion"])
        wandb.log({
            f"{name}/rd_curve":
                wandb.plot.line(table, "rate", "distortion", title="RD Curve")
        })

        # loss_lst = np.array(loss_lst)
        rate_lst = np.array(rate_lst)
        dist_lst = np.array(dist_lst)
        auc_dict = {
            f"{name}/max_rate": np.max(rate_lst),
            f"{name}/min_rate": np.min(rate_lst),
            f"{name}/abs_rate": np.max(rate_lst) - np.min(rate_lst),
            f"{name}/max_dist": np.max(dist_lst),
            f"{name}/min_dist": np.min(dist_lst),
            f"{name}/abs_dist": np.max(dist_lst) - np.min(dist_lst),
        }
        wandb.log(auc_dict)


def hyper_train(model, biq, criterion, optimizer, scheduler, cfg):
    do_checkpoint = cfg.checkpoint_dir is not None
    if do_checkpoint and os.path.exists(
            os.path.join(cfg.checkpoint_dir, "checkpoint.pth")):
        slurm_checkpoint = torch.load(
            os.path.join(cfg.checkpoint_dir, "checkpoint.pth"))
        model.load_state_dict(slurm_checkpoint["state_dict"])
        optimizer.load_state_dict(slurm_checkpoint["optimizer"])
        scheduler.load_state_dict(slurm_checkpoint["scheduler"])
        epoch = slurm_checkpoint["epoch"]
    else:
        epoch = 0

    while epoch < cfg.total_epochs:
        do_evaluate = epoch % cfg.eval_freq == 0
        # do_evaluate = epoch % cfg.eval_freq == 0 and epoch != 0

        do_save = epoch % cfg.save_freq == 0 and epoch != 0

        if do_evaluate:
            hyper_evaluate(model, criterion, epoch, "train_eval")
            hyper_evaluate(model, criterion, epoch, "test")

        if do_checkpoint and do_save:
            slurm_check_dir = os.path.join(cfg.checkpoint_dir, "checkpoint.pth")
            log_info = {
                "id": wandb.run.id,
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            torch.save(log_info, slurm_check_dir)

        model.train()
        loader = biq(args.data_name, "train", cfg.batch_size, DEVICE)
        p_bar = tqdm.tqdm(loader)
        metric_dict = initialize_metric(criterion.get_metric_lst())

        for batch in p_bar:
            optimizer.zero_grad()
            inputs = batch["inputs"]
            output_dict = model.sample_forward(inputs)
            loss, loss_dict = criterion(output_dict)
            loss.backward()
            optimizer.step()

            metric_dict = update_metric(metric_dict, loss_dict, inputs.size(0))
            summ_dict = summarize_metric(metric_dict)
            summ_str = generate_metric_str("train", epoch, summ_dict)
            p_bar.set_description(summ_str)

        summ_dict = summarize_metric(metric_dict, name="train_step/")
        summ_dict["beta"] = cfg.get_beta(epoch)
        wandb.log(summ_dict)
        epoch = epoch + 1
        scheduler.step()

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
    model = build_hyper_model(args.data_name, hyper_cfg, DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler1 = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-5,
        end_factor=1.,
        total_iters=10)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=290)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[10])

    criterion = build_hyper_criterion(DEVICE)

    hyper_train(model, build_input_queue, criterion, optimizer, scheduler, cfg)
    hyper_evaluate(model, criterion, cfg.total_epochs, "train_eval")
    hyper_evaluate(model, criterion, cfg.total_epochs, "test")

    # import matplotlib.pyplot as plt
    # # Visualizing the reconstruction
    # test_loader = build_input_queue("test", cfg.batch_size, DEVICE)
    # test_batch = next(test_loader)
    # outputs_dict = model.forward(test_batch["inputs"])
    # logits = outputs_dict["logits"].view(-1, 28, 28)
    # plt.figure(figsize=(5, 5))
    # plt.axis("square")
    # for i in range(50):
    #     data_i = test_batch["inputs"].view(-1, 28, 28)[i].data.cpu().numpy()
    #     recon_i = torch.sigmoid(logits[i]).data.cpu().numpy()
    #     plt.subplot(10, 10, 2 * i + 1)
    #     plt.imshow(data_i, cmap="Greys")
    #     plt.axis("off")
    #     plt.subplot(10, 10, 2 * i + 2)
    #     plt.imshow(recon_i, cmap="Greys")
    #     plt.axis("off")
    # wandb.log({"reconstruction": plt})
    # wandb.finish()


if __name__ == "__main__":
    main()
