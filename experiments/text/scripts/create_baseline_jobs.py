import argparse

import numpy as np

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_jobs")
parser.add_argument("--experiment_name", type=str, default="hv_text_jobs_final")

args = parser.parse_args()

PTB_CONFIG = {
    "lr": [0.0003],
    "data_name": ["ptb"],
    "total_epochs": [100],
    "decoder_name": ["lstm", "trans"],
    "schedule": ["constant"],
    "beta": list(np.logspace(-2, 1, num=10)),
    "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

YAH_CONFIG1 = {
    "lr": [0.0003],
    "data_name": ["yahoo"],
    "total_epochs": [100],
    "decoder_name": ["lstm"],
    "schedule": ["constant"],
    "beta": list(np.logspace(-2, 1, num=10)),
    "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

YAH_CONFIG2 = {
    "lr": [0.001],
    "data_name": ["yahoo"],
    "total_epochs": [100],
    "decoder_name": ["trans"],
    "schedule": ["constant"],
    "beta": list(np.logspace(-2, 1, num=10)),
    "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      PTB_CONFIG,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      YAH_CONFIG1,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      YAH_CONFIG2,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
