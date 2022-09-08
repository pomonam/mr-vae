import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="bn_sweep")
parser.add_argument(
    "--experiment_name", type=str, default="hvae_bimage_nas_sweep_bn_type_v2")

args = parser.parse_args()


CONFIG = {
    "lr": [3e-3,
           1e-3,
           3e-4,
           1e-4],
    "total_epochs": [200],
    "data_name": ["mnist", "omniglot"],
    "encoder_name": ["conv"],
    "decoder_name": ["conv"],
    "shared_preprocess": [1],
    "layer_type": ["sig_gate"],
    "param_type": ["pre_bn", "post_act"],
    "apply_zero_init": [0],
    "block_type": ["mlp"],
    "norm_type": ["none", "scale_shift"],
    "apply_bn_tracking": [0, 1],
    "apply_bn_calibrate": [0, 1],
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs))
