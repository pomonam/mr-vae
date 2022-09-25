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
  plt.rcParams.update(bundles.iclr2023(rel_width=0.7))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.with_edge())

  beta = [0.01, 0.02154, 0.04642, 0.1, 0.2154, 0.4642, 1, 3.793, 6.158, 10]
  au_wo_annealing = [32, 32, 32, 32, 32, 32, 30, 8, 3, 2]
  au_w_annealing = [32, 32, 32, 32, 32, 32, 19, 15, 4, 2]

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
  res = [32] * 16 + [22, 10, 4, 2]
  plt.xscale("log")

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "192k97si")
  plt.plot(beta, res, "o-", c=rgb.tue_green, linewidth=1.)
  #
  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "3giz7dld")
  # plt.plot(rate, dist, "o-", c=rgb.tue_red, linewidth=1.)

  plt.xlim(0.05, 12)
  plt.ylim(0, 35)

  # plt.xlabel(r"$\beta$")
  plt.ylabel("Active Units")
  plt.title("Omniglot")
  plt.grid()
  plt.tight_layout()
  plt.savefig("au_omniglot.pdf", bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  main()
