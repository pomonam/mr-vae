"""Plot rate-distortion curves from rd_curve_demo.py / baseline_comparison.py output.

Prefers `comparison.csv` (combined MR-VAE + baseline output), falls back to
`rd_curve.csv` (MR-VAE only).

    pip install matplotlib
    python plot_rd_curve.py
"""
import argparse
import csv
import os
import sys


def read_combined(path):
  mr, base = [], []
  with open(path) as f:
    for row in csv.DictReader(f):
      beta, rate, dist = (
          float(row["beta"]),
          float(row["rate"]),
          float(row["distortion"]),
      )
      (mr if row["method"] == "mr_vae" else base).append((beta, rate, dist))
  return mr, base


def read_mr_only(path):
  with open(path) as f:
    return [(float(r["beta"]), float(r["rate"]), float(r["distortion"]))
            for r in csv.DictReader(f)], []


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, default=None,
                      help="CSV to plot. If omitted, uses comparison.csv or rd_curve.csv.")
  parser.add_argument("--output", type=str, default="rd_plot.png")
  parser.add_argument("--title", type=str,
                      default="Rate-Distortion frontier — MR-VAE vs β-VAEs")
  args = parser.parse_args()

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    sys.exit("matplotlib not installed. Run: pip install matplotlib")

  src = args.input
  if src is None:
    src = "comparison.csv" if os.path.exists("comparison.csv") else "rd_curve.csv"
  if not os.path.exists(src):
    sys.exit(f"No data found ({src}). Run rd_curve_demo.py or baseline_comparison.py first.")

  if src.endswith("comparison.csv") or "method" in open(src).readline():
    mr, base = read_combined(src)
  else:
    mr, base = read_mr_only(src)

  fig, ax = plt.subplots(figsize=(7, 5))
  if mr:
    rates = [r for _, r, _ in mr]
    dists = [d for _, _, d in mr]
    ax.plot(rates, dists, "o-", color="C3", label="MR-VAE (single run)",
            linewidth=2, markersize=6)
  if base:
    brates = [r for _, r, _ in base]
    bdists = [d for _, _, d in base]
    ax.scatter(brates, bdists, s=140, color="C0", marker="s",
               label="β-VAE (separate run per β)",
               edgecolors="black", linewidths=1, zorder=5)
    for beta, r, d in base:
      ax.annotate(f"β={beta:g}", (r, d), textcoords="offset points",
                  xytext=(8, 6), fontsize=9, color="C0")

  ax.set_xlabel("Rate (KL, nats)")
  ax.set_ylabel("Distortion (BCE, nats)")
  ax.set_title(args.title)
  ax.grid(alpha=0.3)
  ax.legend(loc="upper right")
  fig.tight_layout()
  fig.savefig(args.output, dpi=120)
  print(f"Saved {args.output} ({len(mr)} MR-VAE pts, {len(base)} β-VAE pts) from {src}")


if __name__ == "__main__":
  main()
