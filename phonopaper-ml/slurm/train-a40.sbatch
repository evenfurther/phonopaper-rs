#!/bin/bash
#
# Train the PhonoPaper detector on one A40 and evaluate the best epoch.
#
# Usage (from the repository root or anywhere):
#
#   sbatch phonopaper-ml/slurm/train-a40.sbatch
#
# Tunables are environment variables, e.g.:
#
#   DATASET=dataset-200k EPOCHS=60 BATCH_SIZE=512 LEARNING_RATE=3e-3 \
#       sbatch phonopaper-ml/slurm/train-a40.sbatch
#
# The job:
#   1. builds phonopaper-train (CUDA backend, no TUI) if needed;
#   2. generates the dataset if $DATASET does not exist yet (CPU-bound);
#   3. trains with early stopping — the best-validation epoch is exported
#      to $ARTIFACTS/model.bin automatically;
#   4. evaluates that model on the validation and training splits.
#
# Logs go to <jobname>-<jobid>.out / .err in the directory you submit from.

#SBATCH --job-name=phonopaper-train
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ─── Configuration (override with environment variables) ─────────────────────

REPO="${REPO:-$HOME/Dev/phonopaper-rs}"
ML="$REPO/phonopaper-ml"

DATASET="${DATASET:-dataset-200k}"        # relative to $ML unless absolute
DATASET_COUNT="${DATASET_COUNT:-200000}"  # only used if the dataset is missing
ARTIFACTS="${ARTIFACTS:-artifacts-$SLURM_JOB_ID}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-2e-3}"
PATIENCE="${PATIENCE:-6}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
SEED="${SEED:-42}"
CUDA_MODULE="${CUDA_MODULE:-cuda}"

# ─── Environment ─────────────────────────────────────────────────────────────

if command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE" || echo "warning: could not load module $CUDA_MODULE" >&2
fi
# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

cd "$ML"

echo "== job $SLURM_JOB_ID on $(hostname), $(date)"
echo "== dataset=$DATASET artifacts=$ARTIFACTS epochs=$EPOCHS batch=$BATCH_SIZE lr=$LEARNING_RATE patience=$PATIENCE workers=$WORKERS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# ─── 1. Build ────────────────────────────────────────────────────────────────

cargo build --release -p phonopaper-dataset
cargo build --release -p phonopaper-train --no-default-features --features cuda
TRAIN=./target/release/phonopaper-train

# ─── 2. Dataset (generated only if missing) ──────────────────────────────────

if [ ! -f "$DATASET/manifest.json" ]; then
    echo "== generating $DATASET ($DATASET_COUNT images)"
    ./target/release/phonopaper-dataset --output "$DATASET" --count "$DATASET_COUNT"
fi

# ─── 3. Train (exports the best-validation epoch to $ARTIFACTS/model.bin) ────

echo "== training, $(date)"
"$TRAIN" train \
    --dataset "$DATASET" --artifacts "$ARTIFACTS" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
    --patience "$PATIENCE" --workers "$WORKERS" --seed "$SEED" \
    | grep -vE 'TrainingProgress'

echo "== mean validation loss per epoch"
for d in "$ARTIFACTS"/valid/epoch-*; do
    printf '%s ' "$(basename "$d")"
    awk -F, '{s+=$1;n++} END {printf "%.4f\n", s/n}' "$d/Loss.log"
done | sort -t- -k2 -n

# ─── 4. Evaluate the exported (best) model on both splits ────────────────────

echo "== evaluation, $(date)"
"$TRAIN" eval --dataset "$DATASET" --artifacts "$ARTIFACTS" --batch-size "$BATCH_SIZE"
"$TRAIN" eval --dataset "$DATASET" --artifacts "$ARTIFACTS" --batch-size "$BATCH_SIZE" --split train

echo "== done, $(date); model: $ML/$ARTIFACTS/model.bin"
