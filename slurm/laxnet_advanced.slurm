#!/bin/bash
# Re-run only the advanced (player-only) stats:
#   sbatch --account=<allocation> slurm/laxnet_advanced.slurm runs/laxnet/<game>
#SBATCH --job-name=lax-advanced
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/laxnet_advanced_%j.out
set -euo pipefail

module purge
module load miniforge
module load ffmpeg/8.0
source activate laxnet 2>/dev/null || conda activate laxnet
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

RUN_DIR=${1:-runs/laxnet/hfx_at_ga_2026-05-02}
python -m laxnet advanced --run "$RUN_DIR"
