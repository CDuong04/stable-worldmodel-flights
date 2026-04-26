#!/usr/bin/env bash
#SBATCH --job-name=cartpole-open-loop
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cartpole-openloop-%j.out
#SBATCH --error=logs/cartpole-openloop-%j.err

# Override knobs:
#   sbatch --export=ALL,WEIGHTS=/oscar/scratch/$USER/stablewm/checkpoints/cartpole_lewm_v2_ln/weights_epoch_50.pt,EPISODES=8,HORIZON=80 \
#          scripts/submit_cartpole_open_loop_check.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_DEFAULT_ENV:-base}"
fi

DEFAULT_SCRATCH="${SCRATCH:-/oscar/scratch/$USER}"
[[ -d "$DEFAULT_SCRATCH" ]] || DEFAULT_SCRATCH="$HOME"
export STABLEWM_HOME="${STABLEWM_HOME:-$DEFAULT_SCRATCH/stablewm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p logs

if [[ -z "${WEIGHTS:-}" ]]; then
  CKPT_DIR="$STABLEWM_HOME/checkpoints/cartpole_lewm_v2_ln"
  WEIGHTS="$(ls -1v "$CKPT_DIR"/weights_epoch_*.pt 2>/dev/null | tail -n 1 || true)"
  if [[ -z "$WEIGHTS" ]]; then
    echo "No weights found in $CKPT_DIR; pass WEIGHTS=<path>" >&2
    exit 1
  fi
fi

EPISODES="${EPISODES:-8}"
HORIZON="${HORIZON:-80}"
START_STEP="${START_STEP:-0}"

echo "WEIGHTS=$WEIGHTS  EPISODES=$EPISODES  HORIZON=$HORIZON  START_STEP=$START_STEP"

python scripts/plan/cartpole_open_loop_check.py \
  --weights "$WEIGHTS" \
  --episodes "$EPISODES" \
  --horizon "$HORIZON" \
  --start-step "$START_STEP"
