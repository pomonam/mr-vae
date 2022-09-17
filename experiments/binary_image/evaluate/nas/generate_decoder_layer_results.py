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
# HYPER_NAME = "hvae_bimage_nas_sweep_decoder_layer_type_v14"
HYPER_NAME = "hvae_bimage_nas_sweep_decoder_layer_type_v20"


def main():
  plt.rcParams.update({"figure.dpi": 300})
  plt.rcParams.update(bundles.neurips2022(ncols=1, nrows=1))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="mnist",
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"Independent Training",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "2v9xttlr")
  plt.plot(rate, dist, "o-", label="Sigmoid Gating", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "32nrbm5g")
  plt.plot(rate, dist, "o-", label="Sqrt Gating", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "8ps7i6ky")
  # plt.plot(rate, dist, "o-", label="Tanh Gating", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "6qw0hpqw")
  plt.plot(rate, dist, "o-", label="Affine Gating", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "2a3ar0iv")
  plt.plot(rate, dist, "o-", label="FiLM Layer", linewidth=1.5)

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
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"Independent Training",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "36lf44zf")
  plt.plot(rate, dist, "o-", label="Sigmoid Gating", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1o89iurd")
  plt.plot(rate, dist, "o-", label="Sqrt Gating", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "2s83tk2j")
  # plt.plot(rate, dist, "o-", label="Tanh Gating", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "5zrhfixk")
  plt.plot(rate, dist, "o-", label="Affine Gating", linewidth=1.5)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "tb1z1yj4")
  plt.plot(rate, dist, "o-", label="FiLM Layer", linewidth=1.5)

  plt.xlim(0, 120)
  plt.ylim(20, 120)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("Omniglot Dataset")
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
