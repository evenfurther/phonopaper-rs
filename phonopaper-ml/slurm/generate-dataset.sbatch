#!/bin/bash
#
# Generate the deterministic PhonoPaper training dataset on the CPU partition.
#
#   cd phonopaper-ml && sbatch slurm/generate-dataset.sbatch
#
# Tunables are environment variables, for example:
#
#   DATASET=dataset-v2-scale3-300k DATASET_COUNT=300000 \
#       sbatch slurm/generate-dataset.sbatch
#
# The generator uses Rayon and this job limits it to the allocated CPU count.

#SBATCH --job-name=phonopaper-dataset
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO="${REPO:-$HOME/Dev/phonopaper-rs}"
ML="$REPO/phonopaper-ml"
# shellcheck source=../scripts/lib.sh
source "$ML/scripts/lib.sh"

DATASET="$(absolute_from_ml "${DATASET:-dataset-v2-scale3-200k}")"
DATASET_COUNT="${DATASET_COUNT:-200000}"
SOURCE_SCALE="${SOURCE_SCALE:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
POSITIVE_RATIO="${POSITIVE_RATIO:-0.6}"
DATASET_SEED="${DATASET_SEED:-1592590337}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# shellcheck disable=SC1091
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
cd "$ML"

CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //; s/[^A-Za-z0-9]\+/-/g; s/-*$//')
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ML/target/cpu-${CPU_MODEL:-unknown}}"
DATASET_BIN=(cargo run --quiet --release -p phonopaper-dataset --)

echo "== job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "== dataset=$DATASET count=$DATASET_COUNT size=$IMAGE_SIZE source_scale=$SOURCE_SCALE threads=$RAYON_NUM_THREADS"

cargo build --release -p phonopaper-dataset
prepare_dataset "$DATASET" "$DATASET_COUNT" "$SOURCE_SCALE" "$IMAGE_SIZE" \
    "$DATASET_SEED" "$POSITIVE_RATIO" "${DATASET_BIN[@]}"

echo "== complete, $(date): $DATASET"
