import argparse
import numpy as np
from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_binary_jobs")
parser.add_argument(
  "--experiment_name",
  type=str,
  default="nvae_binary_train_baseline")

args = parser.parse_args()

CONFIG = {
  "dataset": ["mnist", "omniglot"],
  "kl_fixed": np.logspace(-3, 1, num=20),
}


if __name__ == "__main__":
  base_str = "python new_train.py --data ../../../logs/data --root checkpoints/ --batch_size 200 " \
             "--epochs 400 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 " \
             "--num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 " \
             "--num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 " \
             "--ada_groups --use_se --res_dist --fast_adamax "
  jobs = generate_job_strings(
    CONFIG,
    command_template=base_str + "--experiment_name {} ".format(
      args.experiment_name))
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  # generate_sh_file(args.file_name, len(jobs))
