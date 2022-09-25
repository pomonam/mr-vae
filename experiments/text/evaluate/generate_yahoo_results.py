import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.text.evaluate.utils import get_baseline_rd
from experiments.text.evaluate.utils import get_hyper_rd

ENTITY = "bae-group"
BASELINE_NAME = "hypervae_text_train_v5"
HYPER_NAME = "hypervae_text_hyper_train_v5"


def main():
  plt.rcParams.update({"figure.dpi": 300})
  # plt.rcParams.update(bundles.iclr2023())
  plt.rcParams.update(bundles.iclr2023(rel_width=0.7))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.with_edge())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="yahoo",
                               schedule="cyclic", arc_name="lstm", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    # label=r"LSTM (Retraining)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_mauve,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="yahoo",
                               schedule="constant", arc_name="lstm", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"$\beta$-VAEs",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_mauve,
    marker="v"
  )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1ggsvqez")
  plt.plot(rate, dist, "o-", label="MR-VAEs", c=rgb.tue_mauve, linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1a3huym9")
  # plt.plot(rate, dist, "o-", label="Transformer (Hyper)", linewidth=1.5)

  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1i04ot2m")
  # plt.plot(rate, dist, "o-", label="HC-VAE-2", linewidth=1.5)

  plt.xlim(1, 110)
  plt.ylim(300, 390)
  plt.grid()

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("Yahoo")
  plt.legend()
  plt.tight_layout()
  plt.savefig("yahoo.pdf", bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  main()
