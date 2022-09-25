import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.binary_image.evaluate.utils import get_baseline_rd
from experiments.binary_image.evaluate.utils import get_hyper_rd

ENTITY = "bae-group"
BASELINE_NAME = "hvae_bimage_jobs_final_for_fig1"
HYPER_NAME = "ahvae_bimage_hyper_sweep"


def main():
  plt.rcParams.update({"figure.dpi": 900})
  plt.rcParams.update(bundles.iclr2023(rel_width=0.7))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.with_edge())

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"$\beta$-VAEs",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="constant", arc_name="resnet", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"$\beta$-VAEs (KL Annealing)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="^"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
                               schedule="cyclic", arc_name="resnet", test=True)
  plt.plot([0], [0])
  plt.scatter(
    rate,
    dist,
    label=r"$\beta$-VAEs (Cyclic KL Schedule)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="<"
  )

  # rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="omniglot",
  #                              schedule="monotonic", arc_name="conv", test=True)
  # plt.plot([0], [0])
  # plt.scatter(
  #   rate,
  #   dist,
  #   label=r"Conv (Retraining)",
  #   edgecolors="k",
  #   linewidths=0.5,
  #   c=rgb.tue_lightblue,
  #   marker="^"
  # )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "192k97si")
  plt.plot(rate, dist, "o-", label="MR-VAEs", c=rgb.tue_red, linewidth=1.5)
  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "28h0c6ab")
  # plt.plot(rate, dist, "o-", label="Conv (Hyper)", linewidth=1.5)

  plt.xlim(0, 120)
  plt.ylim(0, 190)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("Rate-Distortion Curve")
  plt.legend()
  plt.grid()
  plt.tight_layout()
  plt.savefig("fig1.pdf",)
  plt.show()


if __name__ == "__main__":
  main()
