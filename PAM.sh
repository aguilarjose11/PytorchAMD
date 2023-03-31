#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=PAM
#SBATCH --chdir=/scratch/joseaguilar/PAM/
#SBATCH --exclusive=user
#SBATCH --error=PAM_exp_%J_stderr.txt
#SBATCH --output=PAM_exp_%J_stdout.txt
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=2
#SBATCH --exclusive=user
#SBATCH --array=0-2
#SBATCH --time=12:00:00

echo start time
date

conda activate torch-gpu
# Load parameters for experiment.
mapfile -t < parameters/params_experiment_50.txt
# graph_emb: avg, p-raw, p-emb
CUDA_VISIBLE_DEVICES=0 python PAM.py ${MAPFILE[@]} --graph_emb 'avg' --new_environments --exp_label 'pseudo_test'
