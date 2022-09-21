#!/bin/bash
#SBATCH -N 1
#SBATCH -J test
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --partition=t4v1,p100,t4v2
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --export=ALL
#SBATCH --array=0-28%28
#SBATCH --output=temp/array-%A_%a.out
#SBATCH -c 4

. $HOME/envs/hvae_env
export PYTHONPATH=$HOME/codes/hyper-vae:$PYTHONPATH

IFS=$'\n' read -d '' -r -a lines < hyper_sweep
cd ..

echo ${lines[SLURM_ARRAY_TASK_ID]} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}
eval ${lines[SLURM_ARRAY_TASK_ID]} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}