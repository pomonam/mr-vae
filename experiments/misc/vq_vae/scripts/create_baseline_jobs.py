import argparse

import numpy as np

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_jobs")
parser.add_argument("--experiment_name", type=str, default="hvqvae_image_jobs_final")

args = parser.parse_args()

CONFIG = {
    "lr": [3e-3],
    "total_epochs": [200],
    "data_name": ["mnist", "celeba"],
    "lamb": list(np.logspace(-2, 1, num=10)),
    # "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs), qos="deadline")
