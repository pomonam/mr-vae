import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_sweep")
parser.add_argument(
    "--experiment_name", type=str, default="hvae_image_hyper_sweep_v100")

args = parser.parse_args()

CONV_CONFIG = {
    "lr": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "total_epochs": [200],
    "data_name": ["cifar", "svhn", "celeba"],
    "arch_name": ["conv"],
    "hyper_config_summary": ["linear_default", "linear_ss", "mlp_default"]
}

RENSET_CONFIG = {
    "lr": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "total_epochs": [200],
    "data_name": ["cifar", "svhn", "celeba"],
    "arch_name": ["resnet"],
    "hyper_config_summary": ["linear_default", "linear_ss", "mlp_default"]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONV_CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      RENSET_CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs) - 1)
