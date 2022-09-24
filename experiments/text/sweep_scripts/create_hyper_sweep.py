import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_sweep")
parser.add_argument(
    "--experiment_name", type=str, default="hvae_text_hyper_sweep_v400")

args = parser.parse_args()

CONFIG = {
    "lr": [3e-2, 1e-2, 3e-3, 1e-3, 3e-4],
    "data_name": ["yahoo"],
    "total_epochs": [100],
    "decoder_name": ["lstm", "trans"],
    "hyper_config_summary": ["linear_default"],
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs), qos="deadline")
