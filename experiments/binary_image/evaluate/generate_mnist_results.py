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
HYPER_NAME = "hvae_bimage_hyper_sweep_v100"


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
    c=rgb.tue_lightblue,
    marker="v"
  )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1xn6cuui")
  plt.plot(rate, dist, "o-", label="HC-VAE-1", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "3r60bxur")
  # plt.plot(rate, dist, "o-", label="HC-VAE-2", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1skahjue")
  # plt.plot(rate, dist, "o-", label="MLP (shared weights)", linewidth=1.5)
  #
  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "27je3kfn")
  # plt.plot(rate, dist, "o-", label="Large MLP", linewidth=1.5)

  plt.xlim(0, 120)
  plt.ylim(15, 120)

  # plt.xlabel("Rate")
  # plt.ylabel("Distortion")
  # plt.title("MNIST Dataset")
  # plt.grid()
  # plt.legend()
  # plt.show()
  # plt.clf()

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="mnist",
                               schedule="monotonic", arc_name="conv", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"Independent Training",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="^"
  )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "135cx5xb")
  plt.plot(rate, dist, "o-", label="HC-VAE-1", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1i04ot2m")
  # plt.plot(rate, dist, "o-", label="HC-VAE-2", linewidth=1.5)

  plt.xlim(0, 120)
  plt.ylim(20, 120)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("MNIST Dataset")
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
