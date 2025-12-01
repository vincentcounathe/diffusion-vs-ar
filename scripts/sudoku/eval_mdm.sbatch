#!/bin/bash
#SBATCH --job-name=critic-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /share/sds/vc383/scratch_diffusion-vs-ar/critic/jobs/%x-%j.out
#SBATCH -e /share/sds/vc383/scratch_diffusion-vs-ar/critic/jobs/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/share/sds/vc383/code/diffusion-vs-ar}"
ENV_PATH="${ENV_PATH:-/share/sds/vc383/envs/diffusion}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/share/sds/vc383/scratch_diffusion-vs-ar/sudoku-mdm/checkpoints/sudoku-mdm-base}"
CRITIC_CKPT="${CRITIC_CKPT:-output/critic/info_gain_critic.pt}"
TEST_DATASET="${CRITIC_TEST_DATASET:-sudoku_test}"
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-output/critic_eval}"
WANDB_ROOT="${WANDB_ROOT:-/share/sds/vc383/scratch_diffusion-vs-ar/critic/wandb}"
HF_ROOT="${HF_ROOT:-/share/sds/vc383/cache/hf}"
TMP_ROOT="${TMP_ROOT:-/share/sds/vc383/tmp}"

mkdir -p "$OUTPUT_ROOT/base" "$OUTPUT_ROOT/info_gain"

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

# Baseline decoding (top-k schedule only)
python src/train_bash.py \
  --stage mdm --overwrite_output_dir \
  --model_name_or_path model_config_tiny \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --dataset "$TEST_DATASET" \
  --do_predict \
  --cutoff_len 164 \
  --diffusion_steps 20 \
  --output_dir "$OUTPUT_ROOT/base" \
  --remove_unused_columns False \
  --decoding_strategy deterministic-linear \
  --topk_decoding True

# Info-gain decoding
python src/train_bash.py \
  --stage mdm --overwrite_output_dir \
  --model_name_or_path model_config_tiny \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --dataset "$TEST_DATASET" \
  --do_predict \
  --cutoff_len 164 \
  --diffusion_steps 20 \
  --output_dir "$OUTPUT_ROOT/info_gain" \
  --remove_unused_columns False \
  --decoding_strategy deterministic-linear \
  --use_info_gain_ordering \
  --critic_checkpoint "$CRITIC_CKPT" \
  --info_gain_alpha ${INFO_GAIN_ALPHA:-0.5} \
  --info_gain_tau_util 0.0 \
  --info_gain_tau_conf 0.0
