import argparse
import numpy as np

from experiments.array_utils import generate_job_strings
from experiments.array_utils import generate_sh_file

parser = argparse.ArgumentParser()
parser.add_argument("--file_name", type=str, default="baseline_sweep")
parser.add_argument(
    "--experiment_name", type=str, default="hvae_nvae_sweep_v1")

args = parser.parse_args()

CONFIG = {
    "dataset": ["mnist", "omniglot"],
    "beta": list(np.logspace(-2, 1, num=4)),
    "seed": [0]
}

DEFAULT = "python baseline_train.py --experiment_name {} --root checkpoints/ --batch_size 256 \
        --epochs 200 --num_latent_scales 2 --num_groups_per_scale 10 --num_postprocess_cells 3 --num_preprocess_cells 3 \
        --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 2 \
        --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_channels_enc 32 --num_channels_dec 32 --num_nf 0 \
        --ada_groups --num_process_per_node 2 --use_se --res_dist --fast_adamax ".format(args.experiment_name)

if __name__ == "__main__":
  jobs = generate_job_strings(
      CONFIG,
      command_template=DEFAULT)
  with open(args.file_name, "w") as f:
    f.writelines(jobs)
  generate_sh_file(
      args.file_name,
      len(jobs),
      cluster_name="q"
  )
