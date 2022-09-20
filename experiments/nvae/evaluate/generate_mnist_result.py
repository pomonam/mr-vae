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
  plt.rcParams.update(bundles.neurips2022(ncols=1, nrows=1))
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  # rate, dist = get_baseline_rd(ENTITY, BASELINE_NAME, data_name="celeba",
  #                              schedule="monotonic", arc_name="resnet", test=True)
  rate_lst = [156.58, 109.272, 29.777, 13.252]
  dist_lst = [0.092, 0.797, 50.728, 78.064]
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


  rate_lst = [
147.49,138.59,129.13,118.81,106.50,
89.06,
32.36,17.04,10.00,
5.439]
  dist_lst = [
0.1539, 0.258, 0.5377, 1.2354,
3.151,
9.203,
51.16,
71.93, 93.966, 124.227]
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
