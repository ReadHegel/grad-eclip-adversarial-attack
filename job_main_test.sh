#!/bin/bash
#
#SBATCH --job-name=grad_eclip_test
#SBATCH --partition=common
#SBATCH --qos=hegel2_common
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00:00
#SBATCH --output=logs/main_test_%j.txt
#SBATCH --error=logs/main_test_%j.txt

set -e

cd /home/ml_seminar_diff/grad-eclip-adversarial-attack

export $(cat .env | xargs)

mkdir -p logs

uv run python -u main_test.py
