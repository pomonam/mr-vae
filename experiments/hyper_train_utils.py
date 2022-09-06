import os

import numpy as np
import torch
import tqdm
import wandb

from src.hyper.norm_layers import calibrate_bn
from experiments.train_utils import build_input_queue
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric


def hyper_evaluate(model, loader, criterion, epoch, name, hyper_cfg, device,
                   delta=0.01, train_loader=None):
  model.eval()

  with torch.no_grad():
    sample_lst = model.get_log_uniform_samples(20)
    mid_point = sample_lst[len(sample_lst) // 2]
    mid_loss = 0

    loss_lst = []
    rate_lst = []
    dist_lst = []
    au_lst = []
    mi_lst = []

    for sample in sample_lst:
      queue = build_input_queue(loader, device)
      p_bar = tqdm.tqdm(queue)

      # We might want to calibrate bn.
      if hyper_cfg.apply_bn_calibrate:
        # Reset all statistics...
        model.apply(calibrate_bn)
        run_one_epoch(model, train_loader, sample, device)
        model.eval()

      metric_dict = initialize_metric(criterion.get_eval_metric_lst())
      means = []

      for batch in p_bar:
        output_dict = model.fixed_forward(batch, sample)
        means.append(output_dict["mu"])

        loss_dict = criterion.eval_forward(
            recon_x=output_dict["reconstruction"],
            x=output_dict["data"],
            mu=output_dict["mu"],
            log_var=output_dict["log_var"],
            z=output_dict["z"],
            beta=1.)
        metric_dict = update_metric(metric_dict,
                                    loss_dict,
                                    batch["data"].size(0))
        summ_dict = summarize_metric(metric_dict)
        summ_str = generate_metric_str(name, epoch, summ_dict)
        p_bar.set_description(summ_str)

      summ_dict = summarize_metric(metric_dict, name="")

      means = torch.cat(means, dim=0)
      au_mean = means.mean(0, keepdim=True)

      au_var = means - au_mean
      ns = au_var.size(0)
      au_var = (au_var**2).sum(dim=0) / (ns - 1)

      if sample == mid_point:
        mid_loss = summ_dict["loss"]

      loss_lst.append(summ_dict["loss"])
      rate_lst.append(summ_dict["rate"])
      dist_lst.append(summ_dict["distortion"])
      au_lst.append((au_var >= delta).sum().item())
      mi_lst.append(summ_dict["mi"])

    wandb.log({
        f"{name}/loss_lst": loss_lst,
        f"{name}/rate_lst": rate_lst,
        f"{name}/dist_lst": dist_lst,
        f"{name}/sample_lst": sample_lst,
        f"{name}/au_lst": au_lst,
        f"{name}/mi_lst": mi_lst,
    })

    rd_data = [[x, y] for (x, y) in zip(rate_lst, dist_lst)]
    table = wandb.Table(data=rd_data, columns=["rate", "distortion"])
    wandb.log({
        f"{name}/rd_curve":
            wandb.plot.line(table, "rate", "distortion", title="RD Curve")
    })
  return mid_loss


def run_one_epoch(model, loader, value, device):
  model.train()
  queue = build_input_queue(loader, device)
  p_bar = tqdm.tqdm(queue)

  for batch in p_bar:
    # Don't need to do anything.
    model.fixed_forward(batch, value)


def hyper_train(model,
                train_loader,
                test_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                cfg,
                hyper_cfg,
                valid_loader=None):
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
    do_evaluate = epoch % cfg.eval_freq == 0 and epoch != 0
    do_save = epoch % cfg.save_freq == 0 and epoch != 0

    if do_evaluate:
      hyper_evaluate(model,
                     train_loader,
                     criterion,
                     epoch,
                     "train_eval",
                     hyper_cfg,
                     device,
                     train_loader=train_loader)
      hyper_evaluate(model, test_loader, criterion, epoch, "test", hyper_cfg, device,
                     train_loader=train_loader)

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
    queue = build_input_queue(train_loader, device)
    p_bar = tqdm.tqdm(queue)
    metric_dict = initialize_metric(criterion.get_metric_lst())

    for batch in p_bar:
      output_dict = model.sample_forward(batch)
      loss_dict = criterion(
          recon_x=output_dict["reconstruction"],
          x=output_dict["data"],
          mu=output_dict["mu"],
          log_var=output_dict["log_var"],
          z=output_dict["z"],
          beta=output_dict["beta"])
      optimizer.zero_grad()
      loss_dict["loss"].backward()
      optimizer.step()

      metric_dict = update_metric(metric_dict, loss_dict, batch["data"].size(0))
      summ_dict = summarize_metric(metric_dict)
      summ_str = generate_metric_str("train", epoch, summ_dict)
      p_bar.set_description(summ_str)

    summ_dict = summarize_metric(metric_dict, name="train_step/")
    summ_dict["lr"] = optimizer.param_groups[0]["lr"]
    wandb.log(summ_dict)
    epoch = epoch + 1

    # We don't use it for now.
    # if "ReduceLROnPlateau" in str(scheduler.__class__):
    #   val_loss = hyper_single_evaluate(model,
    #                                    valid_loader,
    #                                    criterion,
    #                                    epoch,
    #                                    "valid",
    #                                    device)
    #   scheduler.step(val_loss)
    scheduler.step()

    if np.isnan(summ_dict["train_step/loss"]):
      wandb.finish(exit_code=1)
      raise ValueError()


def hyper_predict(model, loader, value, device):
  model.eval()
  queue = build_input_queue(loader, device)
  batch = next(queue)
  batch["data"] = batch["data"][:10]
  model_out = model.fixed_forward(batch, value)
  reconstructions = model_out["reconstruction"].cpu().detach()
  z_enc = model_out["z"]
  z = torch.randn_like(z_enc)
  # net_inputs is already set.
  normal_generation = model.decoder(z)["reconstruction"].detach().cpu()
  return batch["data"], reconstructions, normal_generation
