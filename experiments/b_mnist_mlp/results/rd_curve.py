from experiments.plots import init_plotting
import matplotlib.pyplot as plt
import numpy as np
from experiments.init_wandb import init_api

ENTITY = "bae-group"
EXPERIMENT_NAME = "hv-b_mnist_mlp_train"


def get_summary(config_lst, summary_lst, lr=1e-3):
    beta_to_rate = {}
    beta_to_dist = {}

    for i, c in enumerate(config_lst):
        if c["lr"] == lr:
            beta_to_rate[c["beta"]] = summary_lst[i]["train_eval/rate"]
            beta_to_dist[c["beta"]] = summary_lst[i]["train_eval/distortion"]
    return beta_to_rate, beta_to_dist


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
                 if not k.startswith('_')})
            name_list.append(run.name)

    rate_dict, dist_dict = get_summary(config_list, summary_list)

    dist_dict_sorted = {i: dist_dict[i] for i in rate_dict.keys()}
    keys = rate_dict.keys()
    values = zip(rate_dict.values(), dist_dict_sorted.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    dist = np.array([c[1] for c in combined_dict.values()])
    plt.scatter(rate, dist)

    min_val = min(np.min(rate), np.min(dist)) - 10
    max_val = max(np.max(rate), np.max(dist)) + 10
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Rate")
    plt.ylabel("Distortion")
    plt.show()


if __name__ == "__main__":
    main()