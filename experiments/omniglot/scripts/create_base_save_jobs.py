import argparse

import numpy as np

from experiments.job_arrays import generate_job_strings
from experiments.job_arrays import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="train_save_jobs")
parser.add_argument("--experiment_name", type=str, default="hypervae_omniglot_train_save")

args = parser.parse_args()

CONFIG = {
    "lr": [3e-5],
    "total_epochs": [500],
    "encoder_name": ["cnn"],
    "decoder_name": ["cnn"],
    "schedule": ["constant", "cyclic"],
    "beta": np.logspace(-3, 1, num=20),
    "save_eval_checkpoint": [1]
}

if __name__ == "__main__":
    jobs = generate_job_strings(
        CONFIG,
        command_template="python baseline_train.py --experiment_name {} ".format(
            args.experiment_name))
    with open(args.file_name, "w") as f:
        f.writelines(jobs)
    generate_sh_file(args.file_name, len(jobs))
