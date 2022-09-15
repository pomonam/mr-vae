#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=16GB
#SBATCH --export=ALL
#SBATCH --array=0-56%56
#SBATCH --output=temp/array-%A_%a.out
#SBATCH --cpus-per-task=8

. $HOME/envs/hvae_env
export PYTHONPATH=$HOME/codes/hyper-vae:$PYTHONPATH

IFS=$'\n' read -d '' -r -a lines < decoder_layer_sweep
cd ..

echo ${lines[SLURM_ARRAY_TASK_ID]}
eval ${lines[SLURM_ARRAY_TASK_ID]}