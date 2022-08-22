import argparse

import numpy as np

from experiments.job_arrays import generate_job_strings
from experiments.job_arrays import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="train_jobs")
parser.add_argument(
    "--experiment_name", type=str, default="hypervae_text_train_v1")

args = parser.parse_args()

CONFIG = {
    "lr": [3e-3, 1e-3, 3e-4, 3e-4],
    "data_name": ["yahoo", "yelp"],
    "total_epochs": [100],
    "schedule": ["constant", "cyclic"],
    "beta": np.logspace(-3, 1, num=20) + [0]
}

if __name__ == "__main__":
    jobs = generate_job_strings(
        CONFIG,
        command_template="python baseline_train.py --experiment_name {} ".format(
            args.experiment_name))
    with open(args.file_name, "w") as f:
        f.writelines(jobs)
    generate_sh_file(args.file_name, len(jobs))
