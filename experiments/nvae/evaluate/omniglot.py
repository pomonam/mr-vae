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


  rate_lst = [162.96170043945312,150.3053741455078,138.78036499023438,126.08306121826172,111.16978454589844,89.45860290527344,31.61103057861328,16.824108123779297,9.654109001159668,5.479655742645264]


  dist_lst = [0.4114468991756439,0.5684274435043335,0.9163941144943236,1.7960728406906128,4.068247318267822,11.45923137664795,52.47293472290039,72.6950912475586,95.22553253173828,123.1817855834961]




  # rate, dist = get_hyper_rd(ENTITY, HYPER_NAME, "1ubm70m5")
  plt.plot(rate_lst, dist_lst, "o-", label="ResNet (Hyper)", linewidth=1.5)

  plt.xlabel("Rate")
  plt.ylabel("Distortion")
  plt.title("MNIST Dataset")
  plt.legend(ncol=2)
  plt.savefig("nvae_omnigl.pdf", bbox_inches="tight")

  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
