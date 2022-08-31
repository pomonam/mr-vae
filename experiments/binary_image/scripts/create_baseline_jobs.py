import argparse

import numpy as np

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_jobs")
parser.add_argument("--experiment_name", type=str, default="hvae_b_image_jobs")


args = parser.parse_args()

CONV_CONFIG1 = {
    "lr": [0.0003],
    "total_epochs": [200],
    "data_name": ["mnist"],
    "encoder_name": ["conv"],
    "decoder_name": ["conv"],
    "schedule": ["constant", "monotonic"],
    "beta": np.logspace(-3, 1, num=20) + [1],
    "save_final_checkpoint": [1]
}

CONV_CONFIG2 = {
    "lr": [0.0003],
    "total_epochs": [200],
    "data_name": ["omniglot"],
    "encoder_name": ["conv"],
    "decoder_name": ["conv"],
    "schedule": ["constant", "monotonic"],
    "beta": np.logspace(-3, 1, num=20) + [1],
    "save_final_checkpoint": [1]
}

RENSET_CONFIG1 = {
    "lr": [0.0003],
    "total_epochs": [200],
    "data_name": ["mnist"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "schedule": ["constant", "monotonic"],
    "beta": np.logspace(-3, 1, num=20) + [1],
    "save_final_checkpoint": [1]
}

RENSET_CONFIG2 = {
    "lr": [0.001],
    "total_epochs": [200],
    "data_name": ["omniglot"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "schedule": ["constant", "monotonic"],
    "beta": np.logspace(-3, 1, num=20) + [1],
    "save_final_checkpoint": [1]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONV_CONFIG1,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      CONV_CONFIG2,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs = generate_job_strings(
      RENSET_CONFIG1,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      RENSET_CONFIG2,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
