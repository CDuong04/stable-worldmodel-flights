#!/usr/bin/env bash
#SBATCH --job-name=cartpole-mpc-eval
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cartpole-mpc-%j.out
#SBATCH --error=logs/cartpole-mpc-%j.err

# Override at submit time, e.g.:
#   sbatch \
#     --export=ALL,WEIGHTS=/oscar/scratch/$USER/stablewm/checkpoints/cartpole_lewm_v2_ln/weights_epoch_50.pt,EPISODES=50,HORIZON=15,CANDIDATES=512,LAMBDA_V=1.0,METHODS=standard,lyapunov \
#     scripts/submit_cartpole_mpc_eval.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Re-activate the conda env active at submit time.
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_DEFAULT_ENV:-base}"
  echo "Active conda env: $(conda info --envs | awk '/\*/ {print $1}')"
fi

DEFAULT_SCRATCH="${SCRATCH:-/oscar/scratch/$USER}"
[[ -d "$DEFAULT_SCRATCH" ]] || DEFAULT_SCRATCH="$HOME"
export STABLEWM_HOME="${STABLEWM_HOME:-$DEFAULT_SCRATCH/stablewm}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p logs

# Default checkpoint: pick the highest-numbered weights_epoch in the v2_ln run.
if [[ -z "${WEIGHTS:-}" ]]; then
  CKPT_DIR="$STABLEWM_HOME/checkpoints/cartpole_lewm_v2_ln"
  WEIGHTS="$(ls -1v "$CKPT_DIR"/weights_epoch_*.pt 2>/dev/null | tail -n 1 || true)"
  if [[ -z "$WEIGHTS" ]]; then
    echo "No weights found in $CKPT_DIR; pass WEIGHTS=<path> explicitly." >&2
    exit 1
  fi
fi

EPISODES="${EPISODES:-50}"
HORIZON="${HORIZON:-15}"
CANDIDATES="${CANDIDATES:-512}"
LAMBDA_V="${LAMBDA_V:-1.0}"
MAX_STEPS="${MAX_STEPS:-500}"
NUM_ENVS="${NUM_ENVS:-4}"
METHODS="${METHODS:-standard,lyapunov}"
DATASET_NAME="${DATASET_NAME:-cartpole_expert_worldmodel}"

echo "WEIGHTS=$WEIGHTS"
echo "EPISODES=$EPISODES HORIZON=$HORIZON CANDIDATES=$CANDIDATES"
echo "LAMBDA_V=$LAMBDA_V MAX_STEPS=$MAX_STEPS NUM_ENVS=$NUM_ENVS METHODS=$METHODS"

python scripts/plan/cartpole_mpc.py \
  --weights "$WEIGHTS" \
  --dataset-name "$DATASET_NAME" \
  --episodes "$EPISODES" \
  --horizon "$HORIZON" \
  --candidates "$CANDIDATES" \
  --lambda-v "$LAMBDA_V" \
  --max-steps "$MAX_STEPS" \
  --num-envs "$NUM_ENVS" \
  --methods "$METHODS"
