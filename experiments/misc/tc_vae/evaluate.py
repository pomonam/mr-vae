import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.binary_image.evaluate.utils import get_baseline_rd
from experiments.binary_image.evaluate.utils import get_hyper_rd


ENTITY = "bae-group"
BASELINE_NAME = "hvae_bimage_jobs_final"
HYPER_NAME = "ahvae_bimage_hyper_sweep"


def main():
  plt.rcParams.update({"figure.dpi": 300})
  # plt.rcParams.update(bundles.iclr2023())
  plt.rcParams.update(bundles.iclr2023(rel_width=0.7))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.with_edge())
  # plt.grid()
  plt.grid()

  beta = [0.1, 0.2154, 0.4642, 1, 2.154, 4.642, 10]
  au_w_annealing = np.array([0.06894, 0.05423, 0.05757, 0.03409, 0.4354, 0.1271, 0.3692])
  au_w_annealing2 = np.array([0.02981, 0.02311, 0.03097, 0.02935, 0.08866, 0.3641, 0.2129])
  au_w_annealin3 = np.array([0.01185, 0.03757, 0.1066, 0.06993, 0.1405, 0.5179, 0.1594])
  mean = []
  std = []
  for i in range(len(au_w_annealing)):
    mean.append(np.mean([au_w_annealin3[i], au_w_annealing2[i], au_w_annealing[i]]))
    std.append(np.std([au_w_annealin3[i], au_w_annealing2[i], au_w_annealing[i]]))

  plt.errorbar(
    beta, mean, std, color="k", capsize=5
  )
  plt.scatter(
    beta,
    mean,
    edgecolors="k",
    linewidths=0.5,
    c="k",
    label=r"$\beta$-TCVAEs",
    marker="v"
  )

  beta = np.logspace(-1, 1, num=10, base=10)
  res = [0.2694, 0.35, 0.479, 0.484, 0.481, 0.489, 0.483, 0.484, 0.482, 0.461]
  plt.xscale("log")

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "192k97si")
  plt.plot(beta, res, "o-", c=rgb.tue_darkblue, label=r"MR-VAEs", linewidth=1.)
  #
  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "3giz7dld")
  # plt.plot(rate, dist, "o-", c=rgb.tue_red, linewidth=1.)

  plt.xlim(0.1, 12)
  plt.ylim(0, 0.8)
  plt.legend()
  plt.xlabel(r"$\beta$")
  plt.ylabel("MIG Metric")
  # plt.title("Omniglot")
  plt.tight_layout()
  plt.savefig("mig_metric.pdf", bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  main()
