#!/usr/bin/env bash
#SBATCH --job-name=cartpole-wm-train
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cartpole-train-%j.out
#SBATCH --error=logs/cartpole-train-%j.err

# Tweak resources / hyperparams at submit time without editing the file:
#   sbatch --time=14:00:00 \
#          --export=ALL,EPOCHS=50,BATCH=256,LR=1e-4,LAMBDA=2.0,PROJ_NORM=ln,RUN_NAME=cartpole_lewm_v2_ln \
#          scripts/submit_cartpole_train.sh

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
if [[ ! -d "$DEFAULT_SCRATCH" ]]; then
  DEFAULT_SCRATCH="$HOME"
fi
export STABLEWM_HOME="${STABLEWM_HOME:-$DEFAULT_SCRATCH/stablewm}"
mkdir -p logs

# Optional Hydra overrides via env vars.
HYDRA_OVERRIDES=()
[[ -n "${EPOCHS:-}"    ]] && HYDRA_OVERRIDES+=("trainer.max_epochs=${EPOCHS}")
[[ -n "${BATCH:-}"     ]] && HYDRA_OVERRIDES+=("loader.batch_size=${BATCH}")
[[ -n "${LR:-}"        ]] && HYDRA_OVERRIDES+=("optimizer.lr=${LR}")
[[ -n "${LAMBDA:-}"    ]] && HYDRA_OVERRIDES+=("loss.state_weight=${LAMBDA}")
[[ -n "${PROJ_NORM:-}" ]] && HYDRA_OVERRIDES+=("projector.norm=${PROJ_NORM}")
[[ -n "${WANDB:-}"     ]] && HYDRA_OVERRIDES+=("wandb.enabled=${WANDB}")
[[ -n "${RUN_NAME:-}"  ]] && HYDRA_OVERRIDES+=("output_model_name=${RUN_NAME}")

echo "STABLEWM_HOME=$STABLEWM_HOME"
echo "Hydra overrides: ${HYDRA_OVERRIDES[*]:-(none)}"

python scripts/train/cartpole_wm.py "${HYDRA_OVERRIDES[@]}"
