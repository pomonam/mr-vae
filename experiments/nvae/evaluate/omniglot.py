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
HYPER_NAME = "hvae_nvae_hyper_sweep_v1"


def main():
  plt.rcParams.update({"figure.dpi": 300})
  plt.rcParams.update(bundles.iclr2023(ncols=1, nrows=1))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  # rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="celeba",
  #                              schedule="monotonic", arc_name="resnet", test=True)
  rate_lst = [170.922, 125.119, 42.21, 14.478]
  dist_lst = [0.09585, 0.8476, 51.285, 100.434]
  plt.plot([0], [0])
  plt.scatter(
    rate_lst,
    dist_lst,
    label=r"ResNet (Retraining)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="v"
  )
  rate_lst = [173.618, 122.923, 40.934, 0.2279
]
  dist_lst = [0.1141, 0.7846, 54.296, 171.79]
  plt.plot([0], [0])
  plt.scatter(
    rate_lst,
    dist_lst,
    label=r"ResNet (Retraining)",
    edgecolors="k",
    linewidths=0.5,
    c=rgb.tue_lightblue,
    marker="v"
  )


  rate_lst = [0.6786818504333496,0.8555627465248108,1.3866277933120728,2.7263529300689697,5.999025344848633,17.603029251098633,55.41780471801758,88.71240997314453,141.02406311035156,168.11019897460938]


  dist_lst = [208.5266876220703,188.3874359130859,169.7919921875,151.21974182128906,130.52430725097656,98.5629425048828,45.06983184814453,21.59174346923828,5.075843811035156,0.5635794401168823]




  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1ubm70m5")
  plt.plot(rate_lst, dist_lst, "o-", label="ResNet (Hyper)", linewidth=1.5)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("MNIST Dataset")
  plt.legend(ncol=2)
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
