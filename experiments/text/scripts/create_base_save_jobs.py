import argparse

import numpy as np

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_save_jobs")
parser.add_argument("--experiment_name", type=str, default="hv_text_save_jobs")

args = parser.parse_args()

CONFIG = {
    "lr": [1],
    "total_epochs": [50],
    "data_name": ["yahoo", "yelp"],
    "schedule": ["cyclic", "constant"],
    "beta": np.logspace(-3, 1, num=20),
    "save_final_checkpoint": [1]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
