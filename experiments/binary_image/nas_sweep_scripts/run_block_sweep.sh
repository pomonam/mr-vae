#!/bin/bash
#SBATCH -N 1
#SBATCH -J test
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --partition=t4v1,p100,t4v2,rtx6000
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --array=0-40%40
#SBATCH --output=temp/array-%A_%a.out
#SBATCH -c 4

. $HOME/envs/hvae_env
export PYTHONPATH=$HOME/codes/hyper-vae:$PYTHONPATH

IFS=$'\n' read -d '' -r -a lines < block_sweep
cd ..

echo ${lines[SLURM_ARRAY_TASK_ID]} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}
eval ${lines[SLURM_ARRAY_TASK_ID]} --checkpoint_dir /checkpoint/${USER}/${SLURM_JOB_ID}