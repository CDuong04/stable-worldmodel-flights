#!/usr/bin/env bash
#SBATCH --job-name=cartpole-expert-data
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cartpole-%j.out
#SBATCH --error=logs/cartpole-%j.err

# To override CPU/env counts at submit time without editing the file:
#   sbatch --cpus-per-task=8 --export=ALL,NUM_ENVS=8 scripts/submit_cartpole_collect.sh

# CPU-only alternative (no GPU queue wait): replace the two lines above with:
#   #SBATCH --partition=batch
# and drop the --gres line, then set MUJOCO_GL=osmesa below. Software rendering
# is slower (~3–5x) but needs no GPU.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Prefer user-set STABLEWM_HOME, then $SCRATCH (SLURM/HPC), then Oscar's
# /oscar/scratch/$USER, then $HOME/stablewm as a last resort.
DEFAULT_SCRATCH="${SCRATCH:-/oscar/scratch/$USER}"
if [[ ! -d "$DEFAULT_SCRATCH" ]]; then
  DEFAULT_SCRATCH="$HOME"
fi
export STABLEWM_HOME="${STABLEWM_HOME:-$DEFAULT_SCRATCH/stablewm}"
mkdir -p logs "$STABLEWM_HOME"

NUM_ENVS="${NUM_ENVS:-${SLURM_CPUS_PER_TASK:-4}}"
EPISODES="${EPISODES:-1800}"
DATASET_NAME="${DATASET_NAME:-cartpole_expert_worldmodel}"
echo "STABLEWM_HOME=$STABLEWM_HOME  MUJOCO_GL=$MUJOCO_GL  NUM_ENVS=$NUM_ENVS  EPISODES=$EPISODES"

python scripts/collect_cartpole_expert_data.py \
    --dataset-name "$DATASET_NAME" \
    --episodes "$EPISODES" \
    --num-envs "$NUM_ENVS" \
    --image-size 224 224 \
    --max-episode-steps 500 \
    --seed 7 \
    --noise-std 0.02 \
    --burst-prob 0.01 \
    --burst-noise-std 0.15 \
    --burst-steps 2 5 \
    --vary-visuals \
    --vary-dynamics \
    --video-episodes 12 \
    --chunk-size 64
