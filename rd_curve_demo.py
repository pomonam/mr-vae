"""Reproduce the qualitative behaviour of Figure 1 in the MR-VAE paper.

Trains one MR-VAE on a subset of binarized MNIST for a small number of epochs,
then sweeps β across `n_beta` log-uniformly-spaced values and reports the
resulting (rate, distortion) points. A correctly-trained MR-VAE should trace
out a smooth Pareto frontier (low β → high rate, low distortion; high β → low
rate, high distortion) from a single training run.

Run from the repo root:

    WANDB_MODE=disabled python rd_curve_demo.py

The defaults are tuned for fast CPU runs (~3-5 min). For a fuller comparison
to paper numbers, raise --n_train and --total_epochs.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.binary_image.hyper_train import (
    HyperBinaryImageCriterion,
    build_model,
)
from experiments.binary_image.input_pipeline import load_mnist_data
from src.utils import seed_everything


def _to_tensor(batch, device):
  x = batch[0] if isinstance(batch, (list, tuple)) else batch
  return x.to(device)


def train(model, loader, criterion, optimizer, device, *,
          epochs, max_batches_per_epoch):
  t0 = time.time()
  for epoch in range(epochs):
    model.train()
    losses = []
    for batch_idx, batch in enumerate(loader):
      if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
        break
      x = _to_tensor(batch, device)
      out = model.sample_forward({"data": x})
      loss_dict = criterion(
          recon_x=out["reconstruction"], x=out["data"],
          mu=out["mu"], log_var=out["log_var"], z=out["z"],
          beta=out["beta"])
      optimizer.zero_grad()
      loss_dict["loss"].backward()
      optimizer.step()
      losses.append(loss_dict["loss"].item())
    elapsed = time.time() - t0
    print(f"  epoch {epoch + 1}/{epochs}  "
          f"avg_train_loss={np.mean(losses):.2f}  "
          f"({elapsed:.1f}s elapsed)", flush=True)


def evaluate_rd_curve(model, loader, betas, device, max_batches=None):
  model.eval()
  rows = []
  with torch.no_grad():
    for beta in betas:
      rate_sum = 0.0
      dist_sum = 0.0
      n = 0
      for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
          break
        x = _to_tensor(batch, device)
        out = model.fixed_forward({"data": x}, value=float(beta))
        rate = -0.5 * torch.sum(
            1 + out["log_var"] - out["mu"].pow(2) - out["log_var"].exp(),
            dim=-1)
        dist = F.binary_cross_entropy(
            out["reconstruction"].reshape(x.shape[0], -1),
            x.reshape(x.shape[0], -1),
            reduction="none").sum(dim=-1)
        rate_sum += rate.sum().item()
        dist_sum += dist.sum().item()
        n += x.shape[0]
      print(f"  β = {float(beta):>8.4f}  rate = {rate_sum / n:>8.3f}  "
            f"distortion = {dist_sum / n:>8.3f}", flush=True)
      rows.append((float(beta), rate_sum / n, dist_sum / n))
  return rows


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--total_epochs", type=int, default=5)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--n_train", type=int, default=5000,
                      help="Cap on training examples per epoch (for fast demo). "
                      "Pass -1 to use the full dataset.")
  parser.add_argument("--n_beta", type=int, default=10)
  parser.add_argument("--encoder", type=str, default="conv",
                      choices=["conv", "resnet"])
  parser.add_argument("--decoder", type=str, default="conv",
                      choices=["conv", "resnet"])
  parser.add_argument("--data_path", type=str, default="logs/data")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", type=str, default="rd_curve.csv")
  parser.add_argument("--n_eval_batches", type=int, default=20,
                      help="Cap on number of test batches used to evaluate "
                      "each β (for fast demo). Pass -1 to use the full test set.")
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"device={device}")
  print(f"config: encoder={args.encoder}, decoder={args.decoder}, "
        f"epochs={args.total_epochs}, n_train={args.n_train}, "
        f"batch_size={args.batch_size}, lr={args.lr}, seed={args.seed}")

  seed_everything(args.seed)

  model = build_model(args.encoder, args.decoder, device)
  print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

  os.makedirs(args.data_path, exist_ok=True)
  train_loader = load_mnist_data(
      "train", batch_size=args.batch_size, workers=0,
      data_path=args.data_path)
  test_loader = load_mnist_data(
      "test", batch_size=args.batch_size, workers=0,
      data_path=args.data_path)

  max_batches = (args.n_train // args.batch_size) if args.n_train > 0 else None

  criterion = HyperBinaryImageCriterion()
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  print(f"\nTraining MR-VAE on binarized MNIST "
        f"(β ~ log-Uniform[{model.sample_a}, {model.sample_b}])")
  train(model, train_loader, criterion, optimizer, device,
        epochs=args.total_epochs,
        max_batches_per_epoch=max_batches)

  betas = model.get_log_uniform_samples(args.n_beta)
  print(f"\nRate-Distortion sweep ({args.n_beta} β values, log-spaced "
        f"in [{model.sample_a}, {model.sample_b}]):", flush=True)
  max_eval_batches = (args.n_eval_batches if args.n_eval_batches > 0 else None)
  rows = evaluate_rd_curve(model, test_loader, betas, device,
                           max_batches=max_eval_batches)

  print(f"\n{'β':>10} {'rate (nats)':>12} {'distortion':>12} {'-ELBO':>10}")
  print("-" * 48)
  for beta, rate, dist in rows:
    print(f"{beta:>10.4f} {rate:>12.4f} {dist:>12.4f} {rate + dist:>10.4f}")

  with open(args.output, "w") as f:
    f.write("beta,rate,distortion\n")
    for beta, rate, dist in rows:
      f.write(f"{beta},{rate},{dist}\n")
  print(f"\nSaved RD curve to {args.output}")


if __name__ == "__main__":
  main()
