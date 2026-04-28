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
#          --export=ALL,EPOCHS=50,BATCH=128,LR=5e-5,PROJ_NORM=ln,RUN_NAME=cartpole_lewm_fixed \
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
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p logs

DATASET_NAME="${DATASET_NAME:-cartpole_expert_worldmodel}"
RUN_NAME="${RUN_NAME:-cartpole_lewm_fixed}"
RUN_ID="${RUN_ID:-${RUN_NAME}_${SLURM_JOB_ID:-manual}}"

EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LR="${LR:-5e-5}"
TRAIN_HORIZON="${TRAIN_HORIZON:-5}"
HISTORY_SIZE="${HISTORY_SIZE:-3}"
STATE_REPR="${STATE_REPR:-cossin}"
PROJ_NORM="${PROJ_NORM:-ln}"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-cartpole_lewm}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-1}"
PRECISION="${PRECISION:-bf16}"

LATENT_WEIGHT="${LATENT_WEIGHT:-1.0}"
ROLLOUT_WEIGHT="${ROLLOUT_WEIGHT:-1.0}"
STATE_WEIGHT="${STATE_WEIGHT:-1.0}"
PRED_STATE_WEIGHT="${PRED_STATE_WEIGHT:-1.0}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.02}"

HYDRA_OVERRIDES=(
  "subdir=${RUN_ID}"
  "output_model_name=${RUN_NAME}"
  "data.dataset.name=${DATASET_NAME}"
  "trainer.accelerator=${ACCELERATOR}"
  "trainer.devices=${DEVICES}"
  "trainer.precision=${PRECISION}"
  "trainer.max_epochs=${EPOCHS}"
  "loader.batch_size=${BATCH}"
  "loader.num_workers=${NUM_WORKERS}"
  "optimizer.lr=${LR}"
  "train_horizon=${TRAIN_HORIZON}"
  "wm.history_size=${HISTORY_SIZE}"
  "wm.state_repr=${STATE_REPR}"
  "projector.norm=${PROJ_NORM}"
  "wandb.enabled=${WANDB}"
  "wandb.config.project=${WANDB_PROJECT}"
  "wandb.config.name=${RUN_ID}"
  "loss.latent_weight=${LATENT_WEIGHT}"
  "loss.rollout_weight=${ROLLOUT_WEIGHT}"
  "loss.state_weight=${STATE_WEIGHT}"
  "loss.pred_state_weight=${PRED_STATE_WEIGHT}"
  "loss.sigreg.weight=${SIGREG_WEIGHT}"
)

echo "STABLEWM_HOME=$STABLEWM_HOME"
echo "MUJOCO_GL=$MUJOCO_GL"
echo "DATASET_NAME=$DATASET_NAME"
echo "RUN_NAME=$RUN_NAME"
echo "RUN_ID=$RUN_ID"
echo "ACCELERATOR=$ACCELERATOR DEVICES=$DEVICES PRECISION=$PRECISION"
echo "Hydra overrides: ${HYDRA_OVERRIDES[*]:-(none)}"

python scripts/train/cartpole_wm.py "${HYDRA_OVERRIDES[@]}"
