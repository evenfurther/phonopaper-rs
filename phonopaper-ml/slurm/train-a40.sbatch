#!/bin/bash
#
# Train the PhonoPaper detector on one A40 and evaluate the best epoch.
#
#   cd phonopaper-ml && sbatch slurm/train-a40.sbatch
#
# Tunables are environment variables (see slurm/lib/train-and-eval.sh), e.g.:
#
#   DATASET=dataset-v2-scale3-200k EPOCHS=60 BATCH_SIZE=32 LEARNING_RATE=1e-3 \
#       sbatch slurm/train-a40.sbatch
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

# Conservative defaults for the stride-2 (64×64) heat-map head. Increase only
# after checking peak memory for this backend and GPU.
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"

REPO="${REPO:-$HOME/Dev/phonopaper-rs}"
# shellcheck disable=SC1091
source "$REPO/phonopaper-ml/slurm/lib/train-and-eval.sh"
