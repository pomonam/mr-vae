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
  plt.rcParams.update(bundles.iclr2023(rel_width=0.7))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.with_edge())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="mnist",
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="mnist",
                               schedule="monotonic", arc_name="conv", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_red,
    marker="^"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="mnist",
                               schedule="constant", arc_name="resnet", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="mnist",
                               schedule="constant", arc_name="conv", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_red,
    marker="^"
  )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "107fi8xk")
  plt.plot(rate, dist, "o-", c=rgb.tue_green, linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "xjz7awi9")
  plt.plot(rate, dist, "o-", c=rgb.tue_red, linewidth=1.5)

  plt.xlim(0, 110)
  plt.ylim(20, 110)

  plt.scatter(
    0,
    0,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_dark,
    label=r"$\beta$-VAE",
    marker="^"
  )
  plt.scatter(
    0,
    0,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_dark,
    marker="v"
  )
  plt.scatter(
    0,
    0,
    edgecolors="k",
    linewidths=0.5,
    label=r"\texttt{ConvNet}",
    c=rgb.tue_red,
    marker="o"
  )
  plt.plot(0, 0, "o-", c=rgb.tue_dark, label="MR-VAE", linewidth=1.5)

  plt.scatter(
    0,
    0,
    edgecolors="k",
    linewidths=0.5,
    label=r"\texttt{ResNet}",
    c=rgb.tue_green,
    marker="o"
  )

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("MNIST")
  plt.grid()

  plt.legend(ncol=2)

  plt.tight_layout()
  plt.savefig("../../../figures/mnist.pdf", bbox_inches="tight")
  # plt.show()


if __name__ == "__main__":
  main()
