#!/bin/bash
#SBATCH --job-name=timing_context
#SBATCH --output=logs/slurm_logs/slurm_%j.out
#SBATCH --error=logs/slurm_logs/slurm_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00

module load miniforge
source .venv/bin/activate

python main.py configs/instructed.yaml