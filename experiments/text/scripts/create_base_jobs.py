import argparse

import numpy as np

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_jobs")
parser.add_argument(
    "--experiment_name", type=str, default="hypervae_text_train_baseline-v3")

args = parser.parse_args()

CONFIG = {
    "lr": [1, 0.3, 0.1, 0.03, 0.01],
    "total_epochs": [50],
    "data_name": ["yahoo", "yelp"],
    "schedule": ["cyclic"],
    "beta": [1.]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
