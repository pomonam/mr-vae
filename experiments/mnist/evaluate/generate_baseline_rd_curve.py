from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from experiments.init_wandb import init_api
from src.plotting import init_plotting

ENTITY = "bae-group"
EXPERIMENT_NAME = "hypervae_mnist_train_v5"


def get_summary(config_lst,
                summary_lst,
                lr=1e-3,
                schedule="cyclic",
                encoder_name="cnn",
                decoder_name="cnn"):
    beta_to_rate = {}
    beta_to_dist = {}
    beta_to_elbo = {}

    for i, c in enumerate(config_lst):
        if c["lr"] == lr and c["schedule"] == schedule \
            and c["encoder_name"] == encoder_name \
            and c["decoder_name"] == decoder_name:
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


def get_test_summary(config_lst,
                     summary_lst,
                     lr=1e-3,
                     schedule="cyclic",
                     encoder_name="cnn",
                     decoder_name="cnn"):
    beta_to_rate = {}
    beta_to_dist = {}
    beta_to_elbo = {}

    for i, c in enumerate(config_lst):
        if c["lr"] == lr and c["schedule"] == schedule \
            and c["encoder_name"] == encoder_name \
            and c["decoder_name"] == decoder_name:
            beta_to_rate[c["beta"]] = summary_lst[i]["test/rate"]
            beta_to_dist[c["beta"]] = summary_lst[i]["test/distortion"]
            beta_to_elbo[c["beta"]] = summary_lst[i]["test/loss"]
    sorted_beta_to_rate = dict(
        sorted(beta_to_rate.items(), key=lambda item: item[0]))
    sorted_beta_to_dist = dict(
        sorted(beta_to_dist.items(), key=lambda item: item[0]))
    sorted_beta_to_elbo = dict(
        sorted(beta_to_elbo.items(), key=lambda item: item[0]))
    return sorted_beta_to_rate, sorted_beta_to_dist, sorted_beta_to_elbo


def get_rd(experiment_name, lr, name="cnn", test=False):
    api = init_api()
    runs = api.runs(ENTITY + "/" + experiment_name)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if run.state == "finished":
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")})
            name_list.append(run.name)

    if test:
        rate_dict, dist_dict, elbo_dict = get_test_summary(config_list,
                                                           summary_list,
                                                           schedule="cyclic",
                                                           encoder_name=name,
                                                           decoder_name=name,
                                                           lr=lr)
    else:
        rate_dict, dist_dict, elbo_dict = get_summary(config_list,
                                                      summary_list,
                                                      schedule="cyclic",
                                                      encoder_name=name,
                                                      decoder_name=name,
                                                      lr=lr)
    keys = rate_dict.keys()
    values = zip(rate_dict.values(), dist_dict.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    dist = np.array([c[1] for c in combined_dict.values()])
    return rate, dist


def main():
    init_plotting()

    api = init_api()
    runs = api.runs(ENTITY + "/" + EXPERIMENT_NAME)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if run.state == "finished":
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith("_")})
            name_list.append(run.name)

    rate, dist = get_rd(EXPERIMENT_NAME, lr=1e-3, test=True)
    plt.plot(rate, dist, label="1e-3")

    rate, dist = get_rd(EXPERIMENT_NAME, lr=3e-4, test=True)
    plt.plot(rate, dist, label="3e-4")

    rate, dist = get_rd(EXPERIMENT_NAME, lr=1e-4, test=True)
    plt.plot(rate, dist, label="1e-4")

    rate, dist = get_rd(EXPERIMENT_NAME, lr=3e-5, test=True)
    plt.plot(rate, dist, label="3e-5")

    plt.xlabel("Rate (nats)")
    plt.ylabel("-ELBO (nats)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
