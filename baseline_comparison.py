"""Compare a single MR-VAE training run to N separate β-VAE training runs.

Reads the MR-VAE rate-distortion curve from rd_curve.csv (produced by
rd_curve_demo.py) and trains one baseline β-VAE per requested β value with
the same architecture and training budget. The single MR-VAE run should
trace approximately the same Pareto frontier as the collection of separate
β-VAE runs.

Usage from the repo root:

    WANDB_MODE=disabled python baseline_comparison.py

Output is a side-by-side table and an extended `rd_curve.csv` containing
both MR-VAE and baseline points.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.binary_image.baseline_train import (
    BinaryImageCriterion,
    build_model,
)
from experiments.binary_image.input_pipeline import load_mnist_data
from src.utils import seed_everything


def _to_tensor(batch, device):
  x = batch[0] if isinstance(batch, (list, tuple)) else batch
  return x.to(device)


def train_one_beta_vae(beta, train_loader, device, *, epochs, lr, seed,
                       max_batches_per_epoch, encoder, decoder):
  seed_everything(seed)
  model = build_model(encoder, decoder, device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = BinaryImageCriterion()
  t0 = time.time()
  for _ in range(epochs):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
      if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
        break
      x = _to_tensor(batch, device)
      out = model({"data": x})
      loss_dict = criterion(
          recon_x=out["reconstruction"], x=out["data"],
          mu=out["mu"], log_var=out["log_var"], z=out["z"],
          beta=beta)
      optimizer.zero_grad()
      loss_dict["loss"].backward()
      optimizer.step()
  return model, time.time() - t0


def evaluate(model, loader, device, max_batches=None):
  model.eval()
  rate_sum = 0.0
  dist_sum = 0.0
  n = 0
  with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
      if max_batches and batch_idx >= max_batches:
        break
      x = _to_tensor(batch, device)
      out = model({"data": x})
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
  return rate_sum / n, dist_sum / n


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--betas", type=str, default="0.01,0.1,1.0,2.15,10.0",
                      help="Comma-separated β values to train baselines at.")
  parser.add_argument("--total_epochs", type=int, default=5,
                      help="Should match the MR-VAE budget for a fair comparison.")
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--n_train", type=int, default=3000)
  parser.add_argument("--n_eval_batches", type=int, default=10)
  parser.add_argument("--encoder", type=str, default="resnet")
  parser.add_argument("--decoder", type=str, default="resnet")
  parser.add_argument("--data_path", type=str, default="logs/data")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--mr_vae_csv", type=str, default="rd_curve.csv")
  parser.add_argument("--output", type=str, default="comparison.csv")
  args = parser.parse_args()

  betas = [float(b) for b in args.betas.split(",")]
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"device={device}")
  print(f"baseline β values: {betas}")
  print(f"per-β training budget: {args.total_epochs} epochs × "
        f"{args.n_train} examples = ~{args.total_epochs * args.n_train} updates")

  os.makedirs(args.data_path, exist_ok=True)
  train_loader = load_mnist_data(
      "train", batch_size=args.batch_size, workers=0, data_path=args.data_path)
  test_loader = load_mnist_data(
      "test", batch_size=args.batch_size, workers=0, data_path=args.data_path)

  max_batches = (args.n_train // args.batch_size) if args.n_train > 0 else None
  max_eval = (args.n_eval_batches if args.n_eval_batches > 0 else None)

  results = []
  total_start = time.time()
  for beta in betas:
    print(f"\nTraining baseline β-VAE at β={beta}...", flush=True)
    model, elapsed = train_one_beta_vae(
        beta, train_loader, device,
        epochs=args.total_epochs, lr=args.lr, seed=args.seed,
        max_batches_per_epoch=max_batches,
        encoder=args.encoder, decoder=args.decoder)
    rate, dist = evaluate(model, test_loader, device, max_batches=max_eval)
    print(f"  trained in {elapsed:.1f}s, "
          f"test rate={rate:.3f}, distortion={dist:.3f}", flush=True)
    results.append((beta, rate, dist))

  total_elapsed = time.time() - total_start
  print(f"\nTotal baseline training time: {total_elapsed:.1f}s "
        f"({len(betas)} separate runs)")

  mr_vae_rows = []
  if os.path.exists(args.mr_vae_csv):
    with open(args.mr_vae_csv) as f:
      reader = csv.DictReader(f)
      for row in reader:
        mr_vae_rows.append(
            (float(row["beta"]), float(row["rate"]), float(row["distortion"])))

  print("\n" + "=" * 64)
  print("MR-VAE (single run) vs baseline β-VAEs (one run per β)")
  print("=" * 64)
  if mr_vae_rows:
    print(f"\n{'β':>8}  {'MR-VAE rate':>12}  {'MR-VAE dist':>12}  "
          f"{'β-VAE rate':>11}  {'β-VAE dist':>11}")
    print("-" * 64)
    baseline_by_beta = {round(b, 4): (r, d) for b, r, d in results}
    for beta, mr_rate, mr_dist in mr_vae_rows:
      key = round(beta, 4)
      if key in baseline_by_beta:
        br, bd = baseline_by_beta[key]
        print(f"{beta:>8.4f}  {mr_rate:>12.3f}  {mr_dist:>12.3f}  "
              f"{br:>11.3f}  {bd:>11.3f}")
      else:
        print(f"{beta:>8.4f}  {mr_rate:>12.3f}  {mr_dist:>12.3f}  "
              f"{'—':>11}  {'—':>11}")
  else:
    print("\n(rd_curve.csv not found; run rd_curve_demo.py first for side-by-side)")

  with open(args.output, "w") as f:
    f.write("method,beta,rate,distortion\n")
    for beta, rate, dist in mr_vae_rows:
      f.write(f"mr_vae,{beta},{rate},{dist}\n")
    for beta, rate, dist in results:
      f.write(f"beta_vae,{beta},{rate},{dist}\n")
  print(f"\nSaved combined RD points to {args.output}")


if __name__ == "__main__":
  main()
