import argparse

from experiments.job_arrays import generate_job_strings
from experiments.job_arrays import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_jobs")
parser.add_argument("--experiment_name",
                    type=str,
                    default="hypvae-mnist_mlp_hyper-v4")

args = parser.parse_args()

CONFIG = {
    "lr": [3e-4, 1e-4],
    "total_epochs": [400],
    "encoder_name": ["mlp"],
    "decoder_name": ["mlp"],
    "block_type": ["linear", "mlp"],
    "sample_type": ["beta_log_uniform"],
    "preact_hyper": [0, 1],
    "preprocess_beta": [1],
    "hyper_activation": ["sigmoid", "tanh"],
    "include_layer_norm": [0, 1],
    "include_residual_connection": [0, 1],
}

if __name__ == "__main__":
    jobs = generate_job_strings(
        CONFIG,
        command_template="python hyper_train.py --experiment_name {} ".format(
            args.experiment_name))
    with open(args.file_name, "w") as f:
        f.writelines(jobs)
    generate_sh_file(args.file_name, len(jobs))
