import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.init_wandb import init_api

ENTITY = "bae-group"
BASELINE_NAME = "hypervae_omniglot_train_v5"
HYPER_NAME = "hypervae_mnist_omniglot_train_save"
ID = "2i5r05dz"


def get_summary(summary):
  beta_to_rate = dict(zip(summary["test/sample_lst"], summary["test/au_lst"]))
  beta_to_dist = dict(zip(summary["test/sample_lst"], summary["test/mi_lst"]))
  return beta_to_rate, beta_to_dist


def get_baseline_summary(config_lst,
                         summary_lst,
                         lr=1e-3,
                         schedule="cyclic"):
  beta_to_au = {}
  beta_to_mi = {}

  for i, c in enumerate(config_lst):
    if c["lr"] == lr and c["schedule"] == schedule:
      beta_to_au[c["beta"]] = summary_lst[i]["test/au"]
      beta_to_mi[c["beta"]] = summary_lst[i]["test/mi"]

  beta_to_au = dict(sorted(beta_to_au.items(), key=lambda item: item[0]))
  beta_to_mi = dict(sorted(beta_to_mi.items(), key=lambda item: item[0]))
  return beta_to_au, beta_to_mi


def get_baseline_au_mi(experiment_name, lr, schedule="cyclic"):
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
                                          lr=lr)

  return au_dict, mi_dict


def main():
  plt.rcParams.update({"figure.dpi": 300})
  plt.rcParams.update(bundles.aistats2022())
  plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
  plt.rcParams.update(markers.inverted())

  api = init_api()
  runs = api.runs(ENTITY + "/" + HYPER_NAME)

  summary_list, config_list, name_list = [], [], []
  for run in runs:
    if run.id == ID:
      summary_list.append(run.summary._json_dict)
      config_list.append(
        {k: v for k, v in run.config.items() if not k.startswith('_')})
      name_list.append(run.name)

  au_dict, mi_dict = get_baseline_au_mi(BASELINE_NAME, lr=3e-5, schedule="cyclic")
  plt.scatter(
      au_dict.keys(),
      au_dict.values(),
      label=r"Independent Training",
      edgecolors="k",
      linewidths=0.5,
      c=rgb.tue_lightblue)
  plt.plot([0], [0])
  au_dict, mi_dict = get_summary(summary_list[0])
  plt.plot(
    au_dict.keys(),
    au_dict.values(),
    "o-",
    label="Hypernetwork",
    linewidth=2,
    c=rgb.tue_ocre)

  plt.xlabel("beta")
  plt.ylabel("AU")
  plt.xscale("log")

  plt.title(r"$\beta$ vs. Active Units for Omniglot")
  plt.legend()
  plt.grid()
  plt.show()

  au_dict, mi_dict = get_baseline_au_mi(BASELINE_NAME, lr=3e-5, schedule="cyclic")
  plt.scatter(
      mi_dict.keys(),
      mi_dict.values(),
      label=r"Independent Training",
      edgecolors="k",
      linewidths=0.5,
      c=rgb.tue_lightblue)
  plt.plot([0], [0])
  au_dict, mi_dict = get_summary(summary_list[0])
  plt.plot(
    mi_dict.keys(),
    mi_dict.values(),
    "o-",
    label="Hypernetwork",
    linewidth=2,
    c=rgb.tue_ocre)

  plt.xlabel("beta")
  plt.ylabel("MI")
  plt.xscale("log")
  # plt.ylim(2, 5.5)

  plt.title(r"$\beta$ vs. Mutual Information for Omniglot")
  plt.legend()
  plt.grid()
  plt.show()


if __name__ == "__main__":
  main()
