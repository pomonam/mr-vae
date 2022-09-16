import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="encoder_layer_sweep")
parser.add_argument(
    "--experiment_name",
    type=str,
    default="hvae_bimage_nas_sweep_encoder_layer_type_v20")

args = parser.parse_args()

RESNET_CONFIG = {
    "lr": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "total_epochs": [200],
    "data_name": ["mnist", "omniglot"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "encoder_layer_type": ["sig_gate",
                           "inv_sqrt_gate",
                           "tanh_gate",
                           "scale_shift",
                           "affine"],
    "decoder_layer_type": ["sig_gate"],
    "param_type": ["post_act"],
    "block_type": ["mlp"],
    "shared_preprocess": [1],
}

CONV_CONFIG = {
    "lr": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    "total_epochs": [200],
    "data_name": ["mnist", "omniglot"],
    "encoder_name": ["conv"],
    "decoder_name": ["conv"],
    "encoder_layer_type": ["sig_gate", "sqrt_gate", "tanh_gate", "scale_shift"],
    "decoder_layer_type": ["affine"],
    "block_type": ["linear"],
    "shared_preprocess": [1]
}


if __name__ == "__main__":
  jobs = generate_job_strings(
      RESNET_CONFIG,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  # jobs += ["\n"]
  # jobs += generate_job_strings(
  #     CONV_CONFIG,
  #     command_template="python hyper_train.py --experiment_name {} ".format(
  #         args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs) - 1, qos="deadline")
