import argparse

from experiments.job_arrays import generate_job_strings
from experiments.job_arrays import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_jobs")
parser.add_argument("--experiment_name", type=str, default="hv-b_mnist_mlp_hyper-v2")

args = parser.parse_args()

CONFIG = {
    "lr": [1e-3, 3e-4],
    "param_method": ["mlp", "linear", "residual"],
    "training_method": ["sequential", "simultaneous"],
    "hyper_type": ["mult", "add"]
}

if __name__ == "__main__":
    jobs = generate_job_strings(
        CONFIG,
        command_template=
        "python hyper_train.py --experiment_name {} ".format(args.experiment_name))
    with open(args.file_name, "w") as f:
        f.writelines(jobs)
    generate_sh_file(args.file_name, len(jobs))
