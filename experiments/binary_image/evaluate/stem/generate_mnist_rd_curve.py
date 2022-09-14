import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb
from experiments.binary_image.evaluate.utils import get_baseline_summary, get_summary, get_baseline_rd, get_hyper_rd
from experiments.wandb_utils import init_api

ENTITY = "bae-group"
BASELINE_NAME = "hvae_bimage_jobs_final"
HYPER_NAME = "hvae_bimage_hyper_stem_sweep_v11"
RESNET_ID = "3av9a15n"
CONV_ID = ""


def main():
  plt.rcParams.update({"figure.dpi": 300})
  plt.rcParams.update(bundles.aistats2022(column="full"))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME,
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.plot([0], [0])
  plt.scatter(
      rate,
      dist,
      label=r"Independent Training",
      edgecolors="k",
      linewidths=0.5,
      c=rgb.tue_lightblue)

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME,
                               schedule="monotonic", arc_name="conv", test=True)
  plt.plot([0], [0])
  plt.scatter(
      rate,
      dist,
      label=r"Independent Training",
      edgecolors="k",
      linewidths=0.5,
      c=rgb.tue_lightblue)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, RESNET_ID)
  plt.plot(rate, dist, "o-", label="Hypernetwork", linewidth=2, c=rgb.tue_ocre)

  plt.xlim(0, 120)
  plt.ylim(0, 120)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")

  plt.title("Rate-Distortion Curve for MNIST")
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
