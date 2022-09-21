import matplotlib.pyplot as plt
import numpy as np
from tueplots import bundles
from tueplots import cycler
from tueplots import markers
from tueplots.constants.color import palettes
from tueplots.constants.color import rgb

from experiments.wandb_utils import init_api


def get_summary(summary, test=True):
  if test:
    beta_to_rate = dict(
        zip(summary["test/sample_lst"], summary["test/rate_lst"]))
    beta_to_dist = dict(
        zip(summary["test/sample_lst"], summary["test/dist_lst"]))
    beta_to_elbo = dict(
        zip(summary["test/sample_lst"], summary["test/loss_lst"]))
  else:
    beta_to_rate = dict(
        zip(summary["train_eval/sample_lst"], summary["train_eval/rate_lst"]))
    beta_to_dist = dict(
        zip(summary["train_eval/sample_lst"], summary["train_eval/dist_lst"]))
    beta_to_elbo = dict(
        zip(summary["train_eval/sample_lst"], summary["train_eval/loss_lst"]))
  return beta_to_rate, beta_to_dist, beta_to_elbo


def get_baseline_summary(config_lst,
                         summary_lst,
                         data_name="mnist",
                         schedule="monotonic",
                         arc_name="resnet",
                         test=True):
  beta_to_rate = {}
  beta_to_dist = {}
  beta_to_elbo = {}

  for i, c in enumerate(config_lst):
    if c["schedule"] == schedule and c["data_name"] == data_name:
      if test:
        beta_to_rate[c["beta"]] = summary_lst[i]["test/rate"]
        beta_to_dist[c["beta"]] = summary_lst[i]["test/distortion"]
        beta_to_elbo[c["beta"]] = summary_lst[i]["test/loss"]
      else:
        beta_to_rate[c["beta"]] = summary_lst[i]["train_eval/rate"]
        beta_to_dist[c["beta"]] = summary_lst[i]["train_eval/distortion"]
        beta_to_elbo[c["beta"]] = summary_lst[i]["train_eval/loss"]
  sorted_beta_to_rate = dict(
      sorted(beta_to_rate.items(), key=lambda item: item[0]))
  sorted_beta_to_dist = dict(
      sorted(beta_to_dist.items(), key=lambda item: item[0]))
  sorted_beta_to_elbo = dict(
      sorted(beta_to_elbo.items(), key=lambda item: item[0]))
  return sorted_beta_to_rate, sorted_beta_to_dist, sorted_beta_to_elbo


def get_baseline_rd(entity,
                    experiment_name,
                    data_name="mnist",
                    schedule="constant",
                    arc_name="resnet",
                    test=True):
  api = init_api()
  runs = api.runs(entity + "/" + experiment_name)

  summary_list, config_list, name_list = [], [], []
  for run in runs:
    if run.state == "finished":
      summary_list.append(run.summary._json_dict)
      config_list.append(
          {k: v for k, v in run.config.items() if not k.startswith("_")})
      name_list.append(run.name)

  rate_dict, dist_dict, elbo_dict = get_baseline_summary(config_list,
                                                         summary_list,
                                                         data_name=data_name,
                                                         schedule=schedule,
                                                         arc_name=arc_name,
                                                         test=test)
  keys = rate_dict.keys()
  values = zip(rate_dict.values(), dist_dict.values())
  combined_dict = dict(zip(keys, values))

  rate = np.array([c[0] for c in combined_dict.values()])
  dist = np.array([c[1] for c in combined_dict.values()])
  return rate, dist


def get_hyper_rd(entity, experiment_name, _id):
  api = init_api()
  runs = api.runs(entity + "/" + experiment_name)

  summary_list, config_list, name_list = [], [], []
  for run in runs:
    if run.id == _id:
      summary_list.append(run.summary._json_dict)
      config_list.append(
          {k: v for k, v in run.config.items() if not k.startswith('_')})
      name_list.append(run.name)

  rate_dict, dist_dict, elbo_dict = get_summary(summary_list[0], test=True)
  keys = rate_dict.keys()
  values = zip(rate_dict.values(), dist_dict.values())
  combined_dict = dict(zip(keys, values))
  rate = np.array([c[0] for c in combined_dict.values()])
  dist = np.array([c[1] for c in combined_dict.values()])
  return rate, dist
