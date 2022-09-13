import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_jobs")
parser.add_argument(
    "--experiment_name", type=str, default="hypervae_image_hyper_baseline_v3")

args = parser.parse_args()

CONFIG = {
    "lr": [3e-4, 1e-4, 3e-5],
    "total_epochs": [200],
    "data_name": ["cifar", "svhn", "celeba"],
    "include_shift": [0, 1],
    "include_residual_connection": [0, 1],
    "include_sigmoid_activation": [0, 1],
    "preprocess_beta": [0, 1],
    "include_layer_norm": [0, 1],
    "block_type": ["mlp"],
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
