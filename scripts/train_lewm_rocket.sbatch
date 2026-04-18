#!/bin/bash
#SBATCH --job-name=lewm-rocket
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace|blackwell
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/oscar/home/cduong5/stable-worldmodel-flights/logs/lewm-rocket-%j.out
#SBATCH --error=/oscar/home/cduong5/stable-worldmodel-flights/logs/lewm-rocket-%j.err

set -e
mkdir -p /oscar/home/cduong5/stable-worldmodel-flights/logs

# Checkpoints go to scratch (home quota is full)
export STABLEWM_HOME=/oscar/scratch/cduong5/stablewm
mkdir -p "$STABLEWM_HOME"

# WandB — set your API key, entity, project
# export WANDB_API_KEY=...

source /users/cduong5/miniconda3/etc/profile.d/conda.sh
conda activate flight-world-model
cd /oscar/home/cduong5/stable-worldmodel-flights

echo "=== GPU info ==="
nvidia-smi
echo "=== launching training ==="

python scripts/train/lewm.py data=rocket \
  wandb.config.entity=${WANDB_ENTITY:-bats-swift} \
  wandb.config.project=${WANDB_PROJECT:-rocket-lewm}
