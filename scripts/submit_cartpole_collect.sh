#!/usr/bin/env bash
#SBATCH --job-name=cartpole-expert-data
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cartpole-%j.out
#SBATCH --error=logs/cartpole-%j.err

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
echo "STABLEWM_HOME=$STABLEWM_HOME  MUJOCO_GL=$MUJOCO_GL"

python scripts/collect_cartpole_expert_data.py \
    --dataset-name cartpole_expert_worldmodel \
    --episodes 1800 \
    --num-envs 16 \
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
