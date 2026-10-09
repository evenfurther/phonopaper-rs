#!/bin/bash
#
# Train the PhonoPaper detector on one RTX 6000 Pro and evaluate the best
# epoch.
#
#   cd phonopaper-ml && sbatch slurm/train-rtx6000pro.sbatch
#
# Tunables are environment variables (see slurm/lib/train-and-eval.sh), e.g.:
#
#   DATASET=dataset-v2-scale3-200k EPOCHS=60 BATCH_SIZE=64 LEARNING_RATE=1e-3 \
#       sbatch slurm/train-rtx6000pro.sbatch
#
# Logs go to <jobname>-<jobid>.out / .err in the directory you submit from.
#
# These nodes are driven through the Vulkan-based `wgpu` backend, which only
# needs the graphics driver (no CUDA toolkit).  Pass BACKEND=cuda to use CUDA
# instead — the RTX 6000 Pro is a Blackwell GPU, so that requires a CUDA
# module providing libnvrtc >= 12.8 (e.g. CUDA_MODULE=cuda/12.8).

#SBATCH --job-name=phonopaper-train
#SBATCH --partition=RTX6000PRO
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Conservative defaults for the stride-2 (64×64) heat-map head; increase only
# after measuring peak memory on the selected wgpu/CUDA backend.
export BACKEND="${BACKEND:-wgpu}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"
export WORKERS="${WORKERS:-16}"

REPO="${REPO:-$HOME/Dev/phonopaper-rs}"
# shellcheck disable=SC1091
source "$REPO/phonopaper-ml/slurm/lib/train-and-eval.sh"
