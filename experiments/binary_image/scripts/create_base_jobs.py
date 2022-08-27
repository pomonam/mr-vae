import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_jobs")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="hypervae_binary_image_train_baseline")

args = parser.parse_args()

CONFIG = {
    "lr": [1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "total_epochs": [1000],
    "data_name": ["mnist", "omniglot"],
    "schedule": ["cyclic"],
    "beta": [1.]
}

if __name__ == "__main__":
    jobs = generate_job_strings(
        CONFIG,
        command_template="python baseline_train.py --experiment_name {} "
        .format(args.experiment_name))
    with open(args.file_name, "w") as f:
        f.writelines(jobs)
    generate_sh_file(args.file_name, len(jobs))
