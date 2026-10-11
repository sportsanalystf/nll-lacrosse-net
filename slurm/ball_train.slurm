#!/bin/bash
# Fine-tune the RF-DETR ball detector on the exported COCO tiles.
#   python -m laxnet ball-export --run runs/<game>          # on any machine, after ball-review
#   mkdir -p logs   # once: Slurm needs it before the job starts
#   sbatch --account=<allocation> slurm/ball_train.slurm datasets/ball_<game>
# Weights land in models/ball_rfdetr/checkpoint_best_total.pth (the path configs/laxnet.yaml expects);
# the next laxnet_gpu.slurm run then uses the trained detector automatically.
#SBATCH --job-name=lax-balltrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/balltrain_%j.out

set -euo pipefail
DATASET=${1:?usage: sbatch slurm/ball_train.slurm datasets/ball_<game> [epochs]}
EPOCHS=${2:-}
module purge
module load miniforge
source activate laxnet 2>/dev/null || conda activate laxnet
export HF_HUB_CACHE=${HF_HUB_CACHE:-/scratch/$USER/hf_cache}   # model weights on scratch
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}   # token from `hf auth login`
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m laxnet ball-train --dataset "$DATASET" ${EPOCHS:+--epochs $EPOCHS}
