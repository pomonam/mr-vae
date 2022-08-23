import matplotlib.pyplot as plt
import numpy as np

from experiments.b_mnist.results.rd_curve import get_rd
from experiments.init_wandb import init_api
from src.plotting import init_plotting

ENTITY = "bae-group"
EXPERIMENT_NAME = "hypervae_mnist_hyper_train_v2"
ID = "s51petl8"


def get_summary(summary):
    beta_to_rate = dict(
        zip(summary["train_eval/sample_lst"], summary["train_eval/rate_lst"]))
    beta_to_dist = dict(
        zip(summary["train_eval/sample_lst"], summary["train_eval/dist_lst"]))
    beta_to_elbo = dict(
        zip(summary["train_eval/sample_lst"], summary["train_eval/loss_lst"]))
    return beta_to_rate, beta_to_dist, beta_to_elbo


def main():
    init_plotting()

    api = init_api()
    runs = api.runs(ENTITY + "/" + EXPERIMENT_NAME)

    summary_list, config_list, name_list = [], [], []
    for run in runs:
        if run.state == "finished" and run.id == ID:
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k, v in run.config.items() if not k.startswith('_')})
            name_list.append(run.name)

    rate_dict, dist_dict, elbo_dict = get_summary(summary_list[0])

    keys = rate_dict.keys()
    values = zip(rate_dict.values(), dist_dict.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    dist = np.array([c[1] for c in combined_dict.values()])
    plt.plot(rate, dist, label="Hypernetwork")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")

    # min_val = min(np.min(rate), np.min(dist)) - 10
    # max_val = max(np.max(rate), np.max(dist)) + 10
    # plt.xlim(min_val, max_val)
    # plt.ylim(min_val, max_val)

    plt.xlabel("Rate")
    plt.ylabel("Distortion")

    rate, dist = get_rd("hypervae_mnist_train_v1", lr=1e-3, test=False, name="mlp")
    plt.plot(rate, dist, label=r"Retrain (Cyclic) - $10^{-3}$")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")

    rate, dist = get_rd("hypervae_mnist_train_v1", lr=3e-4, test=False, name="mlp")
    plt.plot(rate, dist, label=r"Retrain (Cyclic) - $30^{-4}$")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")

    rate, dist = get_rd("hypervae_mnist_train_v1", lr=1e-4, test=False, name="mlp")
    plt.plot(rate, dist, label=r"Retrain (Cyclic) - $10^{-4}$")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")
    plt.title("RD-curve on Train Dataset")

    rate, dist = get_rd("hypervae_mnist_train_v1", lr=3e-5, test=False, name="mlp")
    plt.plot(rate, dist, label=r"Retrain (Cyclic) - $30^{-5}$")
    plt.scatter(rate, dist, facecolors="none", edgecolors="k")
    plt.title("RD-curve on Train Dataset")


    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
