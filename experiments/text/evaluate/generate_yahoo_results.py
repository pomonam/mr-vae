import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.text.evaluate.utils import get_baseline_rd
from experiments.text.evaluate.utils import get_hyper_rd

ENTITY = "bae-group"
BASELINE_NAME = "hv_text_baseline"
HYPER_NAME = "hypervae_text_hyper_train_v5"


def main():
  plt.rcParams.update({"figure.dpi": 300})
  plt.rcParams.update(bundles.neurips2022(ncols=1, nrows=1))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="yelp",
                               schedule="monotonic", arc_name="lstm", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"LSTM (Retraining)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="v"
  )

  # rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="yahoo",
  #                              schedule="monotonic", arc_name="trans", test=True)
  # plt.plot([0], [0])
  # plt.scatter(
  #   rate,
  #   dist,
  #   label=r"Transformer (Retraining)",
  #   edgecolors="k",
  #   linewidths=0.5,
  #   c=rgb.tue_lightblue,
  #   marker="^"
  # )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "31vnv86h")
  plt.plot(rate, dist, "o-", label="LSTM (Hyper)", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1a3huym9")
  # plt.plot(rate, dist, "o-", label="Transformer (Hyper)", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1i04ot2m")
  # plt.plot(rate, dist, "o-", label="HC-VAE-2", linewidth=1.5)

  # plt.xlim(0, 200)
  # plt.ylim(0, 110)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("CIFAR-10 Dataset")
  plt.legend(ncol=2)
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
