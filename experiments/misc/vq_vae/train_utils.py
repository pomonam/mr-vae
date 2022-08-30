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


def evaluate(model, loader, epoch, name, device):
  model.eval()

  with torch.no_grad():
    loader = build_input_queue(loader, device)
    p_bar = tqdm.tqdm(loader)
    metric_dict = initialize_metric(["loss", "vq_loss", "recon_loss"])

    for batch in p_bar:
      output_dict = model(batch)
      loss_dict = {
        "loss": output_dict["loss"],
        "vq_loss": output_dict["vq_loss"],
        "recon_loss": output_dict["recon_loss"],
      }
      metric_dict = update_metric(metric_dict, loss_dict, batch["data"].size(0))
      summ_dict = summarize_metric(metric_dict)
      summ_str = generate_metric_str(name, epoch, summ_dict)
      p_bar.set_description(summ_str)

  summ_dict = summarize_metric(metric_dict, name=name + "/")
  wandb.log(summ_dict)


def train(model,
          train_loader,
          test_loader,
          optimizer,
          scheduler,
          device,
          cfg):
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
      evaluate(model, train_loader, epoch, "train_eval", device)
      evaluate(model, test_loader, epoch, "test", device)

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
    metric_dict = initialize_metric(["loss", "vq_loss", "recon_loss"])

    for batch in p_bar:
      output_dict = model(batch)
      loss_dict = {
        "loss": output_dict["loss"],
        "vq_loss": output_dict["vq_loss"],
        "recon_loss": output_dict["recon_loss"],
      }
      optimizer.zero_grad()
      output_dict["loss"].backward()
      optimizer.step()

      metric_dict = update_metric(metric_dict, loss_dict, batch["data"].size(0))
      summ_dict = summarize_metric(metric_dict)
      summ_str = generate_metric_str("train", epoch, summ_dict)
      p_bar.set_description(summ_str)

    summ_dict = summarize_metric(metric_dict, name="train_step/")
    summ_dict["lr"] = optimizer.param_groups[0]["lr"]
    wandb.log(summ_dict)
    epoch = epoch + 1

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
  reconstructions = model_out["recon_x"].cpu().detach()
  z_enc = model_out["z"]
  z = torch.randn_like(z_enc)
  normal_generation = model.decoder(z)["recon_x"].detach().cpu()
  return batch["data"], reconstructions, normal_generation
