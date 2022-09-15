import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb
from experiments.binary_image.evaluate.utils import get_baseline_rd, get_hyper_rd

ENTITY = "bae-group"
BASELINE_NAME = "hvae_bimage_jobs_final"
HYPER_NAME = "hvae_bimage_nas_sweep_bn_type_v14"


def main():
  plt.rcParams.update({"figure.dpi": 300})
  plt.rcParams.update(bundles.neurips2022(ncols=1, nrows=1))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="mnist",
                               schedule="monotonic", arc_name="conv", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"Independent Training",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "f2jexqy4")
  plt.plot(rate, dist, "o-", label="Without BN Transform", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "2dwkqk05")
  plt.plot(rate, dist, "o-", label="With BN Transform", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "193gwtgj")
  plt.plot(rate, dist, "o-", label="With IN Transform", linewidth=1.5)
  plt.xlim(0, 120)
  plt.ylim(15, 120)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("MNIST Dataset")
  plt.grid()
  plt.legend()
  plt.show()
  plt.clf()

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="monotonic", arc_name="conv", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"Independent Training",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "2v9vfunp")
  plt.plot(rate, dist, "o-", label="Without BN Transform", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1zdy1mvs")
  plt.plot(rate, dist, "o-", label="Sqrt Gating", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "2orq3u43")
  plt.plot(rate, dist, "o-", label="With IN Transform", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "110dklki")
  # plt.plot(rate, dist, "o-", label="Affine Transformation", linewidth=1.5)

  plt.xlim(0, 120)
  plt.ylim(20, 120)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("Omniglot Dataset")
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
