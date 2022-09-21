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
  plt.rcParams.update(bundles.iclr2023(rel_width=0.6))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"ResNet (Retraining)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="monotonic", arc_name="conv", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"Conv (Retraining)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="^"
  )

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1qezffsn")
  # plt.plot(rate, dist, "o-", label="ResNet (Hyper)", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "28h0c6ab")
  plt.plot(rate, dist, "o-", label="Conv (Hyper)", linewidth=1.5)

  plt.xlim(0, 120)
  plt.ylim(30, 120)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("Omniglot Dataset")
  # plt.legend(ncol=2)
  plt.grid()
  plt.tight_layout()
  plt.savefig("omniglot.pdf", bbox_inches="tight")


if __name__ == "__main__":
  main()
