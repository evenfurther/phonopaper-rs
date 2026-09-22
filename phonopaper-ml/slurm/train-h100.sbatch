#!/bin/bash
#
# Train the PhonoPaper detector on one H100 and evaluate the best epoch.
#
#   cd phonopaper-ml && sbatch slurm/train-h100.sbatch
#
# Tunables are environment variables (see slurm/lib/train-and-eval.sh), e.g.:
#
#   DATASET=dataset-200k EPOCHS=60 BATCH_SIZE=1024 LEARNING_RATE=4e-3 \
#       sbatch slurm/train-h100.sbatch
#
# Logs go to <jobname>-<jobid>.out / .err in the directory you submit from.
#
# Uses the CUDA backend (Hopper, compute capability 9.0, is supported by any
# CUDA 12 toolkit).  Pass BACKEND=wgpu to use Vulkan instead if the CUDA
# module is unavailable on these nodes.

#SBATCH --job-name=phonopaper-train
#SBATCH --partition=H100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Partition-specific defaults: 80 GB of GPU memory and 64-core nodes allow a
# larger batch (with the learning rate scaled by roughly its square root) and
# more data-loader workers.
export BACKEND="${BACKEND:-cuda}"
export BATCH_SIZE="${BATCH_SIZE:-512}"
export LEARNING_RATE="${LEARNING_RATE:-3e-3}"
export WORKERS="${WORKERS:-16}"

REPO="${REPO:-$HOME/Dev/phonopaper-rs}"
# shellcheck disable=SC1091
source "$REPO/phonopaper-ml/slurm/lib/train-and-eval.sh"
