#!/bin/bash
# Generic Slurm launcher.
# Usage: sbatch slurm_run.sh <script.py> [args...]
#
#SBATCH --job-name=py_job
#SBATCH --partition=common
#SBATCH --qos=hegel2_common
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00:00
#SBATCH --output=logs/slurm_%j.txt
#SBATCH --error=logs/slurm_%j.txt

set -e

export $(cat .env | xargs)

mkdir -p logs

PYTHONUNBUFFERED=1 uv run "$@"
