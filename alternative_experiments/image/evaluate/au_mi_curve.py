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
  plt.grid()

  beta = [0.01, 0.02154, 0.04642, 0.1, 0.2154, 0.4642, 1, 2.154, 4.642, 10]
  au_wo_annealing = [64, 64, 64, 64, 64, 64, 55, 52, 31, 18]
  au_w_annealing = [64, 64, 64, 64, 64, 64, 64, 52, 34, 18]

  plt.scatter(
    beta,
    au_w_annealing,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  plt.scatter(
    beta,
    au_wo_annealing,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  beta = np.logspace(-2, 1, num=20, base=10)
  res = [64] * 17 + [45, 36, 35]
  plt.xscale("log")

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "192k97si")
  plt.plot(beta, res, "o-", c=rgb.tue_green, linewidth=1.)
  #
  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "3giz7dld")
  # plt.plot(rate, dist, "o-", c=rgb.tue_red, linewidth=1.)

  # plt.xlim(0, 110)
  # plt.ylim(30, 120)

  plt.xlabel(r"$\beta$")
  plt.ylabel("AU")
  plt.title("CelebA")
  plt.tight_layout()
  plt.savefig("../../../figures/celeba_au.pdf", bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  main()
