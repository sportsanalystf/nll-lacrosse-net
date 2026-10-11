#!/bin/bash
# LacrosseNet on one GPU. Submit from the repository root:
#   mkdir -p logs   # once: Slurm needs it before the job starts
#   sbatch --account=<allocation> slurm/laxnet_gpu_from_track.slurm runs/laxnet/hfx_at_ga_2026-05-02
# Needs: the conda env from slurm/setup_env.sh, an nllcv run of the same video in runs/nllcv/<game>
# (done at least up to `homography`), and ANTHROPIC_API_KEY for jersey numbers
# (put `export ANTHROPIC_API_KEY=...` in ~/.laxnet_env, chmod 600).
# Ball: uses the trained RF-DETR weights in models/ball_rfdetr/ when they exist, otherwise SAM 3 zero-shot.
#SBATCH --job-name=laxnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/laxnet_%j.out

set -euo pipefail
RUN_DIR=${1:-runs/laxnet/hfx_at_ga_2026-05-02}
mkdir -p logs

module purge
module load miniforge
module load ffmpeg/8.0
source activate laxnet 2>/dev/null || conda activate laxnet
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r
[ -f "$HOME/.laxnet_env" ] && source "$HOME/.laxnet_env"

nvidia-smi --query-gpu=name,memory.total --format=csv
export HF_HUB_CACHE=${HF_HUB_CACHE:-/scratch/$USER/hf_cache}   # model weights on scratch
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}   # token from `hf auth login`

python -m laxnet run --run "$RUN_DIR" --backend sam3 --embedder siglip --from-stage track
