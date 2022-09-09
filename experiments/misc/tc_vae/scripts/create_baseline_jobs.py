import argparse

import numpy as np

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_jobs")
parser.add_argument("--experiment_name", type=str, default="hvae_tcvae_jobs_v1")

args = parser.parse_args()

CONFIG = {
    "beta": list(np.logspace(-3, 1, num=20)) + [1],
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
