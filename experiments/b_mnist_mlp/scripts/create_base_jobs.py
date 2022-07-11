import argparse

import numpy as np

from experiments.job_arrays import generate_job_strings
from experiments.job_arrays import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="train_jobs")
parser.add_argument("--experiment_name", type=str, default="hv-b_mnist_mlp_train")

args = parser.parse_args()

CONFIG = {
    "lr": [1e-3],
    "epochs": [200],
    "schedule": ["constant", "monotonic", "cyclic"],
    "beta": np.logspace(-3, 1, num=20)
}

BASE_CONFIG = {
    "lr": [1e-3],
    "epochs": [200],
    "schedule": ["constant"],
    "beta": [0.]
}

if __name__ == "__main__":
    jobs = generate_job_strings(
        CONFIG,
        command_template=
        "python train.py --experiment_name {} ".format(args.experiment_name))
    with open(args.file_name, "w") as f:
        f.writelines(jobs)
    generate_sh_file(args.file_name, len(jobs))
