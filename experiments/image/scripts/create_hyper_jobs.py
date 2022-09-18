import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_jobs")
parser.add_argument(
    "--experiment_name", type=str, default="hypervae_image_hyper_baseline_v100")

args = parser.parse_args()

CONFIG = {
    "lr": [0.001],
    "total_epochs": [200],
    "data_name": ["celeba"],
    "arch_name": ["resnet"],
    "hyper_config_summary": ["linear_default"],
    "save_final_checkpoint": [1]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs), qos="deadline")
