import argparse

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="hyper_jobs")
parser.add_argument(
    "--experiment_name", type=str, default="hvae_bimage_hyper_jobs_final")

args = parser.parse_args()

CONV_CONFIG1 = {
    "lr": [1e-2],
    "total_epochs": [200],
    "data_name": ["omniglot"],
    "encoder_name": ["conv"],
    "decoder_name": ["conv"],
    "hyper_config_summary": ["lin_bn"],
    "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

CONV_CONFIG2 = {
    "lr": [0.0001],
    "total_epochs": [200],
    "data_name": ["mnist"],
    "encoder_name": ["conv"],
    "decoder_name": ["conv"],
    "hyper_config_summary": ["lin_bn"],
    "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

RENSET_CONFIG1 = {
    "lr": [0.001],
    "total_epochs": [200],
    "data_name": ["omniglot"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "hyper_config_summary": ["lin_bn"],
    "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

RENSET_CONFIG2 = {
    "lr": [0.001],
    "total_epochs": [200],
    "data_name": ["mnist"],
    "encoder_name": ["resnet"],
    "decoder_name": ["resnet"],
    "hyper_config_summary": ["lin_bn"],
    "save_final_checkpoint": [1],
    "seed": [0, 1, 2]
}

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONV_CONFIG1,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      CONV_CONFIG2,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      RENSET_CONFIG1,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  jobs += ["\n"]
  jobs += generate_job_strings(
      RENSET_CONFIG2,
      command_template="python hyper_train.py --experiment_name {} ".format(
          args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(args.file_name, len(jobs), qos="deadline")
