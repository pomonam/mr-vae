import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

from experiments.text.input_pipeline import load_data
from experiments.text.models import LstmDecoder
from experiments.text.models import LstmEncoder
from experiments.train_utils import evaluate
from experiments.train_utils import train
from experiments.wandb_utils import init_wandb
from src.config import TrainConfig
from src.evaluate import generate_metric_str
from src.evaluate import initialize_metric
from src.evaluate import summarize_metric
from src.evaluate import update_metric
from src.models.beta_vae import BetaVAE
from src.models.beta_vae import log_sum_exp
from src.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiment_name", type=str, default="hypervae-text-train")

parser.add_argument("--data_name", type=str, default="yahoo")

parser.add_argument("--total_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1.)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--beta", type=float, default=1.)
parser.add_argument("--schedule", type=str, default="constant")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint_dir", type=str, default=None)
parser.add_argument("--save_final_checkpoint", type=int, default=0)
parser.add_argument("--save_freq", type=int, default=5)
parser.add_argument("--eval_freq", type=int, default=15)
args = parser.parse_args()

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")


class TextCriterion(nn.Module):

  @staticmethod
  def get_metric_lst():
    return ["loss", "rate", "distortion"]

  @staticmethod
  def get_eval_metric_lst():
    return TextCriterion.get_metric_lst() + ["mi"]

  @staticmethod
  def forward(recon_x, x, mu, log_var, z, beta):
    recon_loss = recon_x

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    loss_dict = {
        "loss": (recon_loss + beta * kld).mean(dim=0),
        "distortion": recon_loss.mean(dim=0),
        "rate": kld.mean(dim=0)
    }
    return loss_dict

  @staticmethod
  def eval_forward(recon_x, x, mu, log_var, z, beta):
    recon_loss = recon_x

    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

    # Compute MI as well for eval forward.
    batch_size, nz = mu.size()
    neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 *
                   (1 + log_var).sum(-1)).mean()
    z, mu, log_var = z.unsqueeze(1), mu.unsqueeze(1), log_var.unsqueeze(1)
    var = log_var.exp()
    dev = z - mu
    log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                  0.5 * (nz * math.log(2 * math.pi) + log_var.sum(-1))
    log_qz = log_sum_exp(log_density, dim=1) - math.log(batch_size)
    mi = neg_entropy - log_qz.mean(-1)

    loss_dict = {
        "loss": (recon_loss + beta * kld).mean(dim=0),
        "distortion": recon_loss.mean(dim=0),
        "rate": kld.mean(dim=0),
        "mi": mi
    }
    return loss_dict


def build_criterion(device):
  loss_fnc = TextCriterion()
  return loss_fnc.to(device)


def build_model(vocab_size, device):
  model = BetaVAE(
      encoder=LstmEncoder(vocab_size),
      decoder=LstmDecoder(vocab_size),
  )
  return model.to(device)


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


# def calc_iwnll(model, test_data_batch, args, ns=100):
#   report_nll_loss = 0
#   report_num_words = report_num_sents = 0
#   for id_, i in enumerate(np.random.permutation(len(test_data_batch))):
#     batch_data = test_data_batch[i]
#     batch_size, sent_len = batch_data.size()
#
#     # not predict start symbol
#     report_num_words += (sent_len - 1) * batch_size
#
#     report_num_sents += batch_size
#     if id_ % (round(len(test_data_batch) / 10)) == 0:
#       print('iw nll computing %d0%%' % (id_ / (round(len(test_data_batch) / 10))))
#       sys.stdout.flush()
#
#     loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples, ns=ns)
#
#     report_nll_loss += loss.sum().item()
#
#   nll = report_nll_loss / report_num_sents
#   ppl = np.exp(nll * report_num_sents / report_num_words)
#
#   print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
#   sys.stdout.flush()
#   return nll, ppl


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
    do_evaluate = epoch % cfg.eval_freq == 0 and epoch != 0
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
    metric_dict = initialize_metric(criterion.get_metric_lst())
    p_bar = tqdm.tqdm(np.random.permutation(len(train_loader)))

    for i in p_bar:
      batch = train_loader[i]
      batch = {"data": batch.to(device, non_blocking=True)}
      output_dict = model(batch)
      loss_dict = criterion(
          recon_x=output_dict["reconstruction"],
          x=output_dict["data"],
          mu=output_dict["mu"],
          log_var=output_dict["log_var"],
          z=output_dict["z"],
          beta=cfg.get_beta(epoch))
      optimizer.zero_grad()
      loss_dict["loss"].backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()
      metric_dict = update_metric(metric_dict, loss_dict, batch["data"].size(0))

    summ_dict = summarize_metric(metric_dict, name="train_step/")
    summ_dict["beta"] = cfg.get_beta(epoch)
    summ_dict["lr"] = optimizer.param_groups[0]["lr"]
    wandb.log(summ_dict)
    epoch = epoch + 1

    if "ReduceLROnPlateau" in str(scheduler.__class__):
      val_loss = evaluate(model,
                          valid_loader,
                          criterion,
                          epoch,
                          "valid",
                          device)
      scheduler.step(val_loss)
    else:
      scheduler.step()

    if np.isnan(summ_dict["train_step/loss"]):
      wandb.finish(exit_code=1)
      raise ValueError()


def main():
  init_wandb(
      args.checkpoint_dir, project_name=args.experiment_name, config=vars(args))
  cfg = TrainConfig(args)

  seed_everything(cfg.seed)

  train_loader, vocab = load_data(
      args.data_name, "train", cfg.batch_size, data_path="../../logs/text_data")
  valid_loader, _ = load_data(
      args.data_name, "valid", cfg.batch_size, data_path="../../logs/text_data")
  test_loader, _ = load_data(
      args.data_name, "test", cfg.batch_size, data_path="../../logs/text_data")

  vocab_size = len(vocab)
  model = build_model(vocab_size, DEVICE)
  optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
  criterion = build_criterion(DEVICE)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, patience=2, factor=0.5, cooldown=15)

  train(model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        DEVICE,
        cfg,
        valid_loader)
  evaluate(model,
           train_loader,
           criterion,
           cfg.total_epochs,
           "train_eval",
           DEVICE)
  evaluate(model, test_loader, criterion, cfg.total_epochs, "test", DEVICE)

  if args.save_final_checkpoint is not None:
    save_checkpoint = \
      os.path.join("checkpoints", "base_{}_{}_{}.pth".format(args.data_name, args.beta, args.schedule))
    log_info = {
        "state_dict": model.state_dict(),
    }
    torch.save(log_info, save_checkpoint)

  wandb.finish()


if __name__ == "__main__":
  main()
