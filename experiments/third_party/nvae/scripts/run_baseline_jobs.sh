#!/bin/bash
#SBATCH -N 1
#SBATCH -J test
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --partition=t4v1,p100,t4v2,rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --array=0-80%80
#SBATCH --output=temp/array-%A_%a.out
#SBATCH -c 8

. $HOME/envs/hvae_env
export PYTHONPATH=$HOME/codes/hyper-vae:$PYTHONPATH
export PYTHONPATH=$HOME/codes/hyper-vae/experiments/third_party/nvae:$PYTHONPATH

IFS=$'\n' read -d '' -r -a lines < baseline_jobs
cd ..

echo ${lines[SLURM_ARRAY_TASK_ID]} --root /checkpoints/${USER}/${SLURM_JOB_ID}
eval ${lines[SLURM_ARRAY_TASK_ID]} --root /checkpoints/${USER}/${SLURM_JOB_ID}