import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_sweep")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="hv_text_sweep")

args = parser.parse_args()

CONFIG = {
    "lr": [3e-3, 1e-3, 3e-4],
    "data_name": ["yahoo", "ptb"],
    "decoder_name": ["lstm", "trans"],
    "schedule": ["monotonic"],
    "beta": [1.]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python baseline_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
