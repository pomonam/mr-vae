from experiments.wandb_utils import init_api
import matplotlib.pyplot as plt
ENTITY = "bae-group"

RUN_NAMES = [
    "linear_vae_beta_enclogvar_1e-3",
    "linear_vae_beta_enclogvar_1.623e-3",
    "linear_vae_beta_enclogvar_2.637e-3",
    "linear_vae_beta_enclogvar_4.281e-3",
    "linear_vae_beta_enclogvar_6.952e-3",
    "linear_vae_beta_enclogvar_1.129e-2",
    "linear_vae_beta_enclogvar_1.833e-2",
    "linear_vae_beta_enclogvar_2.976e-2",
    "linear_vae_beta_enclogvar_4.833e-2",
    "linear_vae_beta_enclogvar_7.848e-2",
    "linear_vae_beta_enclogvar_1.274e-1",
    "linear_vae_beta_enclogvar_2.069e-1",
    "linear_vae_beta_enclogvar_3.360e-1",
    "linear_vae_beta_enclogvar_5.456e-1",
    "linear_vae_beta_enclogvar_8.859e-1",
    "linear_vae_beta_enclogvar_1.438",
    "linear_vae_beta_enclogvar_2.336",
    "linear_vae_beta_enclogvar_3.793",
    "linear_vae_beta_enclogvar_6.128",
    "linear_vae_beta_enclogvar_10.00",
]

def analytical_rd_metric(a_rate, a_dist, te_rate, te_dist):
    return ((a_rate - te_rate)**2) + ((a_dist - te_dist)**2)

def get_best_rd_point(config_lst, summary_lst):
    analytical_rate = None
    analytical_dist = None
    train_eval_rate = None
    train_eval_dist = None
    lr = None, None, None, None, None
    diff = float("inf")
    for i, c in enumerate(config_lst):
        a_rate = summary_lst[i]["analytical/rate"]
        a_dist = summary_lst[i]["analytical/dist"]
        te_rate = summary_lst[i]["train_eval/rate"]
        te_dist = summary_lst[i]["train_eval/distortion"]
        new_diff = analytical_rd_metric(a_rate, a_dist, te_rate, te_dist)

        if new_diff < diff:
            diff = new_diff
            analytical_rate = a_rate
            analytical_dist = a_dist
            train_eval_rate = te_rate
            train_eval_dist = te_dist
            lr = c["lr"]
        
    return analytical_rate, analytical_dist, train_eval_rate, train_eval_dist, lr

def get_rd_curve():
    api = init_api()
    a_rate, a_dist, te_rate, te_dist, lr = [], [], [], [], []
    for experiment_name in RUN_NAMES:
        runs = api.runs(ENTITY + "/" + experiment_name)

        summary_list, config_list, name_list = [], [], []
        for run in runs:
            if run.state == "finished":
                summary_list.append(run.summary._json_dict)
                config_list.append(
                    {k: v for k, v in run.config.items() if not k.startswith("_")})
                name_list.append(run.name)

        results = get_best_rd_point(config_list, summary_list)
        a_rate.append(results[0])
        a_dist.append(results[1])
        te_rate.append(results[2])
        te_dist.append(results[3])
        lr.append(results[4])

    return a_rate, a_dist, te_rate, te_dist, lr

if __name__ == "__main__":
    a_rate_list, a_dist_list, te_rate_list, te_dist_list, lr_list = get_rd_curve()
    plt.plot(a_rate_list, a_dist_list, label="analytical", linewidth=2, marker='.')
    plt.plot(te_rate_list, te_dist_list, label="single_train", linewidth=2, marker='.')
    for x, y, lr in zip(te_rate_list, te_dist_list, lr_list):
        plt.annotate(text="lr="+str(lr), xy=(x, y))
    plt.legend(loc="best")
    plt.xlabel("rate")
    plt.ylabel("distortion")
    plt.show()