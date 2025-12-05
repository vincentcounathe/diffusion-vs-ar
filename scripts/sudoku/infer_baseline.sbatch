#!/bin/bash
#SBATCH --job-name=mdm-infer-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-high
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 06:00:00

set -euo pipefail

REPO_DIR="${REPO_DIR:-/share/sds/vc383/code/diffusion-vs-ar}"
ENV_PATH="${ENV_PATH:-/share/sds/vc383/envs/diffusion}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share/sds/vc383/scratch_diffusion-vs-ar/sudoku-mdm/checkpoints/sudoku-mdm-base}"
# TEST_DATASET="${TEST_DATASET:-sudoku_test}"
TEST_DATASET="${TEST_DATASET:-critic_test_shah}"
# OUTPUT_DIR="${OUTPUT_DIR:-output/sudoku_eval/base-topk}"
OUTPUT_DIR="${OUTPUT_DIR:-output/critic_eval_shah/base-topk}"
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

source /share/sds/vc383/miniconda/etc/profile.d/conda.sh || true
conda activate "$ENV_PATH"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

python src/train_bash.py \
  --stage mdm --overwrite_output_dir \
  --model_name_or_path model_config_tiny \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --dataset "$TEST_DATASET" \
  --do_predict \
  --cutoff_len 164 \
  --diffusion_steps 20 \
  --output_dir "$OUTPUT_DIR" \
  --remove_unused_columns False \
  --decoding_strategy deterministic-linear \
  --topk_decoding True \
  --report_to wandb
