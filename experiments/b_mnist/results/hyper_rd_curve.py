import argparse
from src.plotting import init_plotting
import matplotlib.pyplot as plt
import numpy as np
from experiments.init_wandb import init_api
from experiments.b_mnist.results.rd_curve import get_rd

#ENTITY = "bae-group"
#EXPERIMENT_NAME = "hv-b_mnist_mlp_hyper-v8"
#ID = "2fno1gt0"
ENTITY = "lrscheduler114"
EXPERIMENT_NAME = "hv-b_mnist_mlp_hyper"
ID = "1e27dd9j"

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default="hyper_vae-hyper-b_mnist_mlp")
args = parser.parse_args()


def get_summary(summary):
    beta_to_rate = dict(zip(summary["train_eval/beta_lst"], summary["train_eval/rate_lst"]))
    beta_to_dist = dict(zip(summary["train_eval/beta_lst"], summary["train_eval/dist_lst"]))
    beta_to_elbo = dict(zip(summary["train_eval/beta_lst"], summary["train_eval/loss_lst"]))
    try:
        beta_to_rate_analytical = dict(zip(summary["analytical/beta_lst"], summary["analytical/rate_lst"]))
        beta_to_dist_analytical = dict(zip(summary["analytical/beta_lst"], summary["analytical/dist_lst"]))
        return beta_to_rate, beta_to_dist, beta_to_elbo, beta_to_rate_analytical, beta_to_dist_analytical
    except:
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
                {k: v for k, v in run.config.items()
                 if not k.startswith('_')})
            name_list.append(run.name)

    results = get_summary(summary_list[0])
    rate_dict = results[0]
    dist_dict = results[1]
    elbo_dict = results[2]
    analytical_rate_dict = results[3] if len(results) == 5 else None
    analytical_dist_dict = results[4] if len(results) == 5 else None

    keys = rate_dict.keys()
    values = zip(rate_dict.values(), dist_dict.values())
    combined_dict = dict(zip(keys, values))

    rate = np.array([c[0] for c in combined_dict.values()])
    dist = np.array([c[1] for c in combined_dict.values()])
    rd = plt.scatter(rate, dist)

    min_val = min(np.min(rate), np.min(dist))
    max_val = max(np.max(rate), np.max(dist))

    if analytical_rate_dict != None and analytical_dist_dict != None:
        keys = analytical_rate_dict.keys()
        values = zip(analytical_rate_dict.values(), analytical_dist_dict.values())
        analytical_combined_dict = dict(zip(keys, values))

        analytical_rate = np.array([c[0] for c in analytical_combined_dict.values()])
        analytical_dist = np.array([c[1] for c in analytical_combined_dict.values()])
        analytical_rd = plt.scatter(analytical_rate, analytical_dist, )
        min_val = min(min_val, np.min(analytical_rate), np.min(analytical_dist))
        max_val = min(max_val, np.max(analytical_rate), np.max(analytical_dist))

    min_val -= 10
    max_val += 10

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.xlabel("Rate")
    plt.ylabel("Distortion")

    rate, dist = get_rd("hv-b_mnist_mlp_train-v5")
    plt.scatter(rate, dist)
    if analytical_rate_dict != None and analytical_dist_dict != None:
        plt.legend((rd, analytical_rd), ('Hyper', 'Analytical'))
    else:
        plt.legend((rd), ('Hyper'))
    plt.show()


if __name__ == "__main__":
    main()