import matplotlib.pyplot as plt
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.image.evaluate.utils import get_baseline_rd
from experiments.image.evaluate.utils import get_hyper_rd


ENTITY = "bae-group"
BASELINE_NAME = "hvae_image_jobs_resnet_final"
HYPER_NAME = "ahvae_image_hyper_sweep"


def main():
  plt.rcParams.update({"figure.dpi": 300})
  # plt.rcParams.update(bundles.iclr2023())
  plt.rcParams.update(bundles.iclr2023(rel_width=0.7))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.with_edge())
  plt.grid()

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="celeba",
                               schedule="monotonic", arc_name="resnet", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="celeba",
                               schedule="monotonic", arc_name="conv", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_red,
    marker="^"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="celeba",
                               schedule="constant", arc_name="resnet", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_green,
    marker="v"
  )

  rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="celeba",
                               schedule="constant", arc_name="conv", test=True)
  plt.scatter(
    rate,
    dist,
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_red,
    marker="^"
  )

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "2txmvy6c")
  plt.plot(rate, dist, "o-", c=rgb.tue_green, linewidth=1.)

  rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "77xxrb50")
  plt.plot(rate, dist, "o-", c=rgb.tue_red, linewidth=1.)

  plt.xlim(0, 200)
  plt.ylim(25, 230)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("CelebA")
  plt.tight_layout()
  plt.savefig("../../../figures/celeba.pdf", bbox_inches="tight")
  plt.show()


if __name__ == "__main__":
  main()
