#!/bin/bash
#SBATCH --partition=swat_plus
#SBATCH --job-name=exp_hw4
#SBATCH --chdir=/scratch/joseaguilar/Hw4/
#SBATCH --exclusive=user
#SBATCH --error=amlhw4_exp_%J_stderr.txt
#SBATCH --output=amlhw4_exp_%J_stdout.txt
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=10
#SBATCH --exclusive=user
#SBATCH --array=0-2
#SBATCH --time=03:00:00

echo start time
date

conda activate torch-gpu
# Load parameters for experiment.
mapfile -t < params.txt

CUDA_VISIBLE_DEVICES=0 python PAM.py ${MAPFILE[@]}