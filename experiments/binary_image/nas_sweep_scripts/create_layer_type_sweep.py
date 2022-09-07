import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="layer_sweep")
parser.add_argument(
    "--experiment_name", type=str, default="hvae_bimage_nas_sweep_layer_type_v3")

args = parser.parse_args()


CONFIG = {
    "lr": [3e-3,
           1e-3,
           3e-4,
           1e-4],
    "total_epochs": [200, 400],
    "data_name": ["mnist", "omniglot"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "shared_preprocess": [0],
    "layer_type": ["sig_gate", "tanh_gate", "scale_shift"],
    "param_type": ["post_act"],
    "apply_zero_init": [0, 1],
    "block_type": ["linear"],
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs), cluster_name="q")
