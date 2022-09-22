import matplotlib.pyplot as plt
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
  plt.rcParams.update(bundles.iclr2023(rel_width=0.8))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.with_edge())
  plt.grid()

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="monotonic", arc_name="conv", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_red,
    marker="^"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="constant", arc_name="resnet", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="constant", arc_name="conv", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_red,
    marker="^"
  )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "192k97si")
  plt.plot(rate, dist, "o-", c=rgb.tue_green, linewidth=1.)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "3giz7dld")
  plt.plot(rate, dist, "o-", c=rgb.tue_red, linewidth=1.)

  plt.xlim(0, 120)
  plt.ylim(30, 120)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("Omniglot")
  plt.tight_layout()
  # plt.savefig("../../../figures/omniglot.pdf", bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  main()
