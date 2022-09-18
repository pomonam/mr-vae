#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p ml
#SBATCH --mem=40GB
#SBATCH --export=ALL
#SBATCH --array=0-8%8
#SBATCH --output=temp/array-%A_%a.out
#SBATCH --nodelist=sonata1
#SBATCH --cpus-per-task=16

. $HOME/envs/hvae_env
export PYTHONPATH=$HOME/codes/hyper-vae:$PYTHONPATH
export PYTHONPATH=$HOME/codes/hyper-vae/experiments/nvae:$PYTHONPATH

IFS=$'\n' read -d '' -r -a lines < hyper_sweep
cd ..

echo ${lines[SLURM_ARRAY_TASK_ID]} --root checkpoints/${SLURM_JOB_ID}
eval ${lines[SLURM_ARRAY_TASK_ID]} --root checkpoints/${SLURM_JOB_ID}