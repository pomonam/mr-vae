import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_sweep")
parser.add_argument(
    "--experiment_name", type=str, default="hvae_image_hyper_sweep")

args = parser.parse_args()

# CONV_CONFIG = {
#     "lr": [3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
#     "total_epochs": [200],
#     "data_name": ["cifar", "svhn", "celeba"],
#     "arch_name": ["conv"],
#     "schedule": ["monotonic"],
#     "beta": [1.]
# }

RENSET_CONFIG = {
    "lr": [1e-3, 3e-4],
    "total_epochs": [200],  # "data_name": ["cifar", "svhn", "celeba"],
    "data_name": ["svhn", "celeba"],
    "arch_name": ["resnet"],
    "include_shift": [0, 1],
    "include_residual_connection": [0, 1],
    "include_sigmoid_activation": [0, 1],
    "preprocess_beta": [0, 1],
    "include_layer_norm": [0, 1],
    "include_output_stem": [0, 1],
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      RENSET_CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  # jobs += ["\n"]
  # jobs += generate_job_strings(
  #     RENSET_CONFIG,
  #     command_template="python baseline_train.py --experiment_name {} ".format(
  #         args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
