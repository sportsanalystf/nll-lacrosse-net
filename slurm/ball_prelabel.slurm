#!/bin/bash
# SAM 3 ball proposals for the labelling frames (adds to the motion proposals from ball-sample).
#   mkdir -p logs   # once: Slurm needs it before the job starts
#   sbatch --account=<allocation> slurm/ball_prelabel.slurm runs/laxnet/hfx_at_ga_2026-05-02
# Afterwards copy runs/<game>/labels/ to your Mac and run `python -m laxnet ball-review --run runs/<game>`.
#SBATCH --job-name=lax-prelabel
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prelabel_%j.out

set -euo pipefail
RUN_DIR=${1:-runs/laxnet/hfx_at_ga_2026-05-02}
module purge
module load miniforge
source activate laxnet 2>/dev/null || conda activate laxnet
export HF_HUB_CACHE=${HF_HUB_CACHE:-/scratch/$USER/hf_cache}   # model weights on scratch
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}   # token from `hf auth login`

# frames + motion proposals first, if not sampled yet
[ -f "$RUN_DIR/labels/index.csv" ] || python -m laxnet ball-sample --run "$RUN_DIR"
python -m laxnet ball-prelabel --run "$RUN_DIR"
