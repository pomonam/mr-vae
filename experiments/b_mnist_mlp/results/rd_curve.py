from experiments.plots import init_plotting
import matplotlib.pyplot as plt
import numpy as np
from experiments.init_wandb import init_api
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

ENTITY = "bae-group"
EXPERIMENT_NAME = "hv-b_mnist_mlp_train"


def get_summary(config_lst, summary_lst, lr=1e-3, schedule="constant"):
    beta_to_rate = {}
    beta_to_dist = {}
    beta_to_elbo = {}

    for i, c in enumerate(config_lst):
        if c["lr"] == lr and c["schedule"] == schedule:
            beta_to_rate[c["beta"]] = summary_lst[i]["train_eval/rate"]
            beta_to_dist[c["beta"]] = summary_lst[i]["train_eval/distortion"]
            beta_to_elbo[c["beta"]] = summary_lst[i]["train_eval/loss"]
    sorted_beta_to_rate = dict(sorted(beta_to_rate.items(), key=lambda item: item[0]))
    sorted_beta_to_dist = dict(sorted(beta_to_dist.items(), key=lambda item: item[0]))
    sorted_beta_to_elbo = dict(sorted(beta_to_elbo.items(), key=lambda item: item[0]))
    return sorted_beta_to_rate, sorted_beta_to_dist, sorted_beta_to_elbo


def main():
    init_plotting()

    api = init_api()
    runs = api.runs(ENTITY + "/" + EXPERIMENT_NAME)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if run.state == "finished":
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items()
                 if not k.startswith("_")})
            name_list.append(run.name)

    rate_dict, dist_dict, elbo_dict = get_summary(config_list,
                                                  summary_list,
                                                  schedule="cyclic")

    keys = rate_dict.keys()
    values = zip(rate_dict.values(), dist_dict.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    dist = np.array([c[1] for c in combined_dict.values()])

    points = np.array([rate, dist]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots()
    keys = np.array(list(keys))
    norm = colors.LogNorm(keys.min(), keys.max())
    lc = LineCollection(segments, cmap="Dark2", norm=norm)
    lc.set_array(keys)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    plt.scatter(rate, dist, facecolors="none", edgecolors="k")

    min_val = min(np.min(rate), np.min(dist)) - 10
    max_val = max(np.max(rate), np.max(dist)) + 10
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Rate")
    plt.ylabel("Distortion")
    plt.tight_layout()
    plt.savefig("rd_curve.pdf")
    plt.clf()

    keys = rate_dict.keys()
    values = zip(rate_dict.values(), elbo_dict.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    elbo = np.array([c[1] for c in combined_dict.values()])

    points = np.array([rate, elbo]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots()
    keys = np.array(list(keys))
    norm = colors.LogNorm(keys.min(), keys.max())
    lc = LineCollection(segments, cmap="Dark2", norm=norm)
    lc.set_array(keys)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    plt.scatter(rate, elbo, facecolors="none", edgecolors="k")

    plt.xlabel("Rate (nats)")
    plt.ylabel("-ELBO (nats)")
    plt.tight_layout()
    plt.savefig("rate_elbo_curve.pdf")


if __name__ == "__main__":
    main()