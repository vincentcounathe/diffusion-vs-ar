#!/bin/bash
#SBATCH --job-name=critic-train
#SBATCH --partition=gpu
# SBATCH --exclude=portal-compute-02
# SBATCH --exclude=elor-compute-02
# SBATCH --constraint=gpu-high
# SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
# SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:TERM@180
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o /share/sds/vc383/scratch_diffusion-vs-ar/critic/jobs/%x-%j.out
#SBATCH -e /share/sds/vc383/scratch_diffusion-vs-ar/critic/jobs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/share/sds/vc383/code/diffusion-vs-ar}"
ENV_PATH="${ENV_PATH:-/share/sds/vc383/envs/diffusion}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share/sds/vc383/scratch_diffusion-vs-ar/sudoku-mdm/checkpoints/sudoku-mdm-base}"
OUTPUT_DIR="${CRITIC_OUTPUT_DIR:-output/critic}"
DATASET="${CRITIC_DATASET:-sudoku_train}"
WANDB_ROOT="${WANDB_ROOT:-/share/sds/vc383/scratch_diffusion-vs-ar/critic/wandb}"
HF_ROOT="${HF_ROOT:-/share/sds/vc383/cache/hf}"
TMP_ROOT="${TMP_ROOT:-/share/sds/vc383/tmp}"

mkdir -p "$OUTPUT_DIR" "$WANDB_ROOT" "$HF_ROOT" "$TMP_ROOT"

if [[ -z "${WANDB_API_KEY:-}" && -f "$HOME/.secrets/wandb_api_key" ]]; then
  export WANDB_API_KEY="$(cat "$HOME/.secrets/wandb_api_key")"
fi
export WANDB_DIR="$WANDB_ROOT"
export WANDB_PROJECT="${WANDB_PROJECT:-diffusion-vs-ar}"
export WANDB__SERVICE_WAIT=300

export HF_HOME="$HF_ROOT"
export TRANSFORMERS_CACHE="$HF_ROOT"
export TMPDIR="$TMP_ROOT"
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

source /share/sds/vc383/miniconda/etc/profile.d/conda.sh || true
conda activate "$ENV_PATH"

wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1 || true

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

python training/train_critic.py \
  --stage mdm \
  --model_name_or_path model_config_tiny \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --dataset "$DATASET" \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --cutoff_len 164 \
  --diffusion_steps 20 \
  --val_size ${CRITIC_VAL_SIZE:-1000} \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size ${CRITIC_BATCH_SIZE:-16} \
  --per_device_eval_batch_size ${CRITIC_EVAL_BATCH_SIZE:-16} \
  --gradient_accumulation_steps ${CRITIC_GRAD_ACC:-1} \
  --learning_rate ${CRITIC_LR:-1e-4} \
  --num_train_epochs ${CRITIC_EPOCHS:-20} \
  --logging_steps 10 \
  --log_interval ${CRITIC_LOG_INTERVAL:-50} \
  --save_steps 1000 \
  --report_to wandb
