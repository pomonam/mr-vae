import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="block_sweep")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="hvae_bimage_nas_sweep_block_type_v14")

args = parser.parse_args()

LINEAR_CONFIG = {
    "lr": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "total_epochs": [200],
    "data_name": ["omniglot"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "encoder_layer_type": ["sig_gate"],
    "decoder_layer_type": ["sqrt_gate"],
    "shared_preprocess": [0],
    "block_type": ["linear"],
}

MLP_CONFIG = {
    "lr": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "total_epochs": [200],
    "data_name": ["mnist", "omniglot"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "encoder_layer_type": ["sig_gate"],
    "decoder_layer_type": ["sqrt_gate"],
    "shared_preprocess": [0, 1],
    "block_type": ["mlp", "large_mlp"],
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      MLP_CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      LINEAR_CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs) - 1, cluster_name="q")
