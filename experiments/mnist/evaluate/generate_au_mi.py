import matplotlib.pyplot as plt
import numpy as np
from tueplots import cycler
from tueplots import cycler, markers
from tueplots.constants.color import palettes
from tueplots import bundles
from experiments.init_wandb import init_api
from tueplots.constants import markers as marker_constants
from tueplots.constants.color import palettes

ENTITY = "bae-group"
BASELINE_NAME = "hypervae_mnist_train_v5"
HYPER_NAME = "hypervae_mnist_hyper_train_save"
ID = "11fjrn2i"


def get_summary(summary, test=True):
    beta_to_rate = dict(
      zip(summary["test/sample_lst"], summary["test/au_lst"]))
    beta_to_dist = dict(
      zip(summary["test/sample_lst"], summary["test/mi_lst"]))
    return beta_to_rate, beta_to_dist


def get_baseline_summary(config_lst,
                         summary_lst,
                         lr=1e-3,
                         schedule="cyclic",
                         test=False):
  beta_to_au = {}
  beta_to_mi = {}

  for i, c in enumerate(config_lst):
    if c["lr"] == lr and c["schedule"] == schedule:

      beta_to_au[c["beta"]] = summary_lst[i]["test/au"]
      beta_to_mi[c["beta"]] = summary_lst[i]["test/mi"]

  beta_to_au = dict(
    sorted(beta_to_au.items(), key=lambda item: item[0]))
  beta_to_mi = dict(
    sorted(beta_to_mi.items(), key=lambda item: item[0]))
  return beta_to_au, beta_to_mi


def get_baseline_rd(experiment_name, lr, schedule="cyclic", test=False):
  api = init_api()
  runs = api.runs(ENTITY + "/" + experiment_name)

  summary_list, config_list, name_list = [], [], []
  for run in runs:
    if run.state == "finished":
      summary_list.append(run.summary._json_dict)
      config_list.append(
        {k: v for k, v in run.config.items() if not k.startswith("_")})
      name_list.append(run.name)

  au_dict, mi_dict = get_baseline_summary(config_list,
                                                         summary_list,
                                                         schedule=schedule,
                                                         lr=lr,
                                                         test=test)


  return au_dict, mi_dict


def main():
  plt.rcParams.update({"figure.dpi": 150})
  # plt.rcParams.update(
  #     cycler.cycler(marker=marker_constants.o_sized[:5], color=palettes.pn[:5])
  # )
  plt.rcParams.update(bundles.aistats2022())
  plt.rcParams.update(cycler.cycler(color=palettes.high_contrast))
  plt.rcParams.update(markers.with_edge())

  api = init_api()
  runs = api.runs(ENTITY + "/" + HYPER_NAME)

  au_dict, mi_dict = get_baseline_rd(BASELINE_NAME, lr=3e-5, schedule="cyclic", test=True)
  plt.scatter(au_dict.keys(), au_dict.values())
  plt.plot([0], [0])

  summary_list, config_list, name_list = [], [], []
  for run in runs:
    if run.id == ID:
      summary_list.append(run.summary._json_dict)
      config_list.append(
        {k: v for k, v in run.config.items() if not k.startswith('_')})
      name_list.append(run.name)

  rate_dict, dist_dict = get_summary(summary_list[0], test=True)
  keys = rate_dict.keys()
  values = zip(rate_dict.values(), dist_dict.values())
  combined_dict = dict(zip(keys, values))
  rate = np.array([c[0] for c in combined_dict.values()])
  dist = np.array([c[1] for c in combined_dict.values()])
  plt.plot(rate_dict.keys(), rate_dict.values(), "o-", label="Hypernetwork", linewidth=2)
  # plt.scatter(rate, dist, facecolors="none", edgecolors="k")

  # rate_dict, dist_dict, elbo_dict = get_summary(summary_list[0], test=False)
  # keys = rate_dict.keys()
  # values = zip(rate_dict.values(), dist_dict.values())
  # combined_dict = dict(zip(keys, values))
  # rate = np.array([c[0] for c in combined_dict.values()])
  # dist = np.array([c[1] for c in combined_dict.values()])
  # plt.plot(rate, dist, "o-", label="Hypernetwork", linewidth=2)
  # plt.scatter(rate, dist, facecolors="none", edgecolors="k")

  # plt.xlim(0, 140)
  # plt.ylim(25, 140)

  plt.xlabel("beta")
  plt.ylabel("AU")
  plt.yscale("log")

  plt.title("Active units for MNIST")
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
