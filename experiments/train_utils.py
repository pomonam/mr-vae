import os

import numpy as np
import torch
import tqdm
import wandb

from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric


def build_input_queue(loader, device):
  for batch in loader:
    if isinstance(batch, list):
      yield {"data": batch[0].to(device, non_blocking=True)}
    else:
      yield {"data": batch.to(device, non_blocking=True)}


def evaluate(model, loader, criterion, epoch, name, device, delta=0.01):
  model.eval()

  with torch.no_grad():
    loader = build_input_queue(loader, device)
    p_bar = tqdm.tqdm(loader)
    metric_dict = initialize_metric(criterion.get_eval_metric_lst())
    means = []

    for batch in p_bar:
      output_dict = model(batch)
      means.append(output_dict["mu"])

      loss_dict = criterion.eval_forward(
          recon_x=output_dict["reconstruction"],
          x=output_dict["data"],
          mu=output_dict["mu"],
          log_var=output_dict["log_var"],
          z=output_dict["z"],
          beta=1.)
      metric_dict = update_metric(metric_dict, loss_dict, batch["data"].size(0))
      summ_dict = summarize_metric(metric_dict)
      summ_str = generate_metric_str(name, epoch, summ_dict)
      p_bar.set_description(summ_str)

  means = torch.cat(means, dim=0)
  au_mean = means.mean(0, keepdim=True)

  au_var = means - au_mean
  ns = au_var.size(0)
  au_var = (au_var**2).sum(dim=0) / (ns - 1)

  summ_dict = summarize_metric(metric_dict, name=name + "/")
  summ_dict[name + "/" + "au"] = (au_var >= delta).sum().item()
  wandb.log(summ_dict)
  return metric_dict["loss"].avg


def train(model,
          train_loader,
          test_loader,
          criterion,
          optimizer,
          scheduler,
          device,
          cfg,
          valid_loader=None):
  do_checkpoint = cfg.checkpoint_dir is not None
  if do_checkpoint and os.path.exists(
      os.path.join(cfg.checkpoint_dir, "checkpoint.pth")):
    slurm_checkpoint = torch.load(
        os.path.join(cfg.checkpoint_dir, "checkpoint.pth"))
    model.load_state_dict(slurm_checkpoint["state_dict"])
    optimizer.load_state_dict(slurm_checkpoint["optimizer"])
    scheduler.load_state_Dict(slurm_checkpoint["scheduler"])
    epoch = slurm_checkpoint["epoch"]
  else:
    epoch = 0

  while epoch < cfg.total_epochs:
    do_evaluate = epoch % cfg.eval_freq == 0
    do_save = epoch % cfg.save_freq == 0 and epoch != 0

    if do_evaluate:
      evaluate(model, train_loader, criterion, epoch, "train_eval", device)
      evaluate(model, test_loader, criterion, epoch, "test", device)

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
      output_dict = model(batch)
      loss_dict = criterion(
          recon_x=output_dict["reconstruction"],
          x=output_dict["data"],
          mu=output_dict["mu"],
          log_var=output_dict["log_var"],
          z=output_dict["z"],
          beta=cfg.get_beta(epoch)
      )
      optimizer.zero_grad()
      loss_dict["loss"].backward()
      optimizer.step()

      metric_dict = update_metric(metric_dict, loss_dict, batch["data"].size(0))
      summ_dict = summarize_metric(metric_dict)
      summ_str = generate_metric_str("train", epoch, summ_dict)
      p_bar.set_description(summ_str)

    summ_dict = summarize_metric(metric_dict, name="train_step/")
    summ_dict["beta"] = cfg.get_beta(epoch)
    summ_dict["lr"] = optimizer.param_groups[0]["lr"]
    wandb.log(summ_dict)
    epoch = epoch + 1

    if "ReduceLROnPlateau" in str(scheduler.__class__):
      val_loss = evaluate(model, valid_loader, criterion, epoch, "valid", device)
      scheduler.step(val_loss)
    else:
      scheduler.step()

    if np.isnan(summ_dict["train_step/loss"]):
      wandb.finish(exit_code=1)
      raise ValueError()


def predict(model, loader, device):
  model.eval()
  queue = build_input_queue(loader, device)
  batch = next(queue)
  batch["data"] = batch["data"][:10]
  model_out = model(batch)
  reconstructions = model_out["reconstruction"].cpu().detach()
  z_enc = model_out["z"]
  z = torch.randn_like(z_enc)
  normal_generation = model.decoder(z)["reconstruction"].detach().cpu()
  return batch["data"], reconstructions, normal_generation
