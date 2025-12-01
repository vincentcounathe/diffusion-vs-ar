#!/bin/bash
#SBATCH --job-name=mdm-sudoku-base-denoiser
#SBATCH --partition=gpu
#SBATCH --exclude=portal-compute-02
#SBATCH --exclude=elor-compute-02
#SBATCH --constraint=gpu-high
# SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:8
#SBATCH --gres=gpu:8
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /share/sds/vc383/scratch_diffusion-vs-ar/jobs/%x-%j.out
#SBATCH -e /share/sds/vc383/scratch_diffusion-vs-ar/jobs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/share/sds/vc383/code/diffusion-vs-ar}"
ENV_PATH="${ENV_PATH:-/share/sds/vc383/envs/diffusion}"
SAVE_ROOT="${SAVE_ROOT:-/share/sds/vc383/scratch_diffusion-vs-ar/sudoku-mdm}"
WANDB_ROOT="${WANDB_ROOT:-$SAVE_ROOT/wandb}"
HF_ROOT="${HF_ROOT:-/share/sds/vc383/cache/hf}" # shared huggingface cache
TMP_ROOT="${TMP_ROOT:-/share/sds/vc383/tmp}"
RUN_NAME="${RUN_NAME:-sudoku-mdm}"
DATASET="${DATASET:-sudoku_train}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXP_DIR="$SAVE_ROOT/checkpoints/${RUN_NAME}-${SLURM_JOB_ID:-local}-$TIMESTAMP"
mkdir -p "$EXP_DIR" "/share/sds/vc383/scratch_diffusion-vs-ar/jobs" "$WANDB_ROOT" "$HF_ROOT" "$TMP_ROOT"

trap 'rsync -a --ignore-existing "$EXP_DIR" "$SAVE_ROOT/checkpoints_backup_${SLURM_JOB_ID}/" || true' EXIT

if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.secrets/wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_api_key")"
fi
export WANDB_DIR="$WANDB_ROOT"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusion-vs-ar}"
export WANDB__SERVICE_WAIT=300
export WANDB_SILENT=true


export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
export HF_HOME="$HF_ROOT"
export TRANSFORMERS_CACHE="$HF_ROOT"
export TMPDIR="$TMP_ROOT"

source /share/sds/vc383/miniconda/etc/profile.d/conda.sh || true
conda activate "$ENV_PATH"

wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

GPUS=$(nvidia-smi -L | wc -l | tr -d ' ')
MASTER_PORT=$((20000 + (RANDOM % 20000)))
export MDM_EXP_DIR="$EXP_DIR"
export MDM_DATASET="$DATASET"
export MDM_NUM_PROCESSES="$GPUS"
export MDM_MASTER_PORT="$MASTER_PORT"
export MDM_EVAL_DATASETS="${EVAL_DATASETS:-sudoku_test}"

bash scripts/sudoku/train-mdm.sh | tee "$EXP_DIR/train.log"
