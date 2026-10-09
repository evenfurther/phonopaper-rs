#!/usr/bin/env bash
# Generate/reuse a compatible dataset, train, then evaluate valid and train splits.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

DATASET=$(absolute_from_ml "${DATASET:-dataset-v2-scale3-50k}")
ARTIFACTS=$(absolute_from_ml "${ARTIFACTS:-artifacts-local}")
DATASET_COUNT=${DATASET_COUNT:-50000}
SOURCE_SCALE=${SOURCE_SCALE:-3}
IMAGE_SIZE=${IMAGE_SIZE:-128}
POSITIVE_RATIO=${POSITIVE_RATIO:-0.6}
DATASET_SEED=${DATASET_SEED:-1592590337}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
PATIENCE=${PATIENCE:-8}
WORKERS=${WORKERS:-4}
SEED=${SEED:-42}
MIN_LR_FRACTION=${MIN_LR_FRACTION:-0.05}
BACKEND=${BACKEND:-flex}

case "$BACKEND" in
    flex) TRAIN_FEATURE_ARGS=() ;;
    cuda | wgpu) TRAIN_FEATURE_ARGS=(--no-default-features --features "$BACKEND") ;;
    *) echo "error: BACKEND must be flex, cuda, or wgpu, got '$BACKEND'" >&2; exit 2 ;;
esac

cd -- "$ML_ROOT"
DATASET_BIN=(cargo run --quiet --release -p phonopaper-dataset --)
TRAIN=(cargo run --quiet --release -p phonopaper-train "${TRAIN_FEATURE_ARGS[@]}" --)

prepare_dataset "$DATASET" "$DATASET_COUNT" "$SOURCE_SCALE" "$IMAGE_SIZE" \
    "$DATASET_SEED" "$POSITIVE_RATIO" "${DATASET_BIN[@]}"

mkdir -p -- "$(dirname -- "$ARTIFACTS")"
echo "== training: backend=$BACKEND dataset=$DATASET artifacts=$ARTIFACTS batch=$BATCH_SIZE"
"${TRAIN[@]}" train --dataset "$DATASET" --artifacts "$ARTIFACTS" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
    --patience "$PATIENCE" --workers "$WORKERS" --seed "$SEED" \
    --input-size "$IMAGE_SIZE" --min-lr-fraction "$MIN_LR_FRACTION"

echo "== evaluating validation split"
"${TRAIN[@]}" eval --dataset "$DATASET" --artifacts "$ARTIFACTS" \
    --batch-size "$BATCH_SIZE" --split valid
echo "== evaluating training split"
"${TRAIN[@]}" eval --dataset "$DATASET" --artifacts "$ARTIFACTS" \
    --batch-size "$BATCH_SIZE" --split train

echo "== complete: $ARTIFACTS/model.bpk"
