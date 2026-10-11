#!/bin/bash
# One-time environment setup on UVA HPC (Rivanna/Afton). Run on a login node:
#   bash slurm/setup_env.sh
# Then request access to the SAM 3 weights at https://huggingface.co/facebook/sam3 and log in once:
#   conda activate laxnet && hf auth login
set -euo pipefail

module purge
module load miniforge            # UVA's conda module; `module spider conda` lists alternatives

ENV=laxnet
if ! conda env list | grep -q "^${ENV} "; then
  conda create -y -n $ENV python=3.12
fi
source activate $ENV 2>/dev/null || conda activate $ENV

# PyTorch with its own CUDA runtime (no CUDA module needed); SAM 3 wants torch >= 2.7, CUDA >= 12.6
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "setuptools<82"     # SAM 3 still imports pkg_resources, removed in setuptools 82

# SAM 3 from source
mkdir -p "$HOME/src"
if [ ! -d "$HOME/src/sam3" ]; then
  git clone https://github.com/facebookresearch/sam3.git "$HOME/src/sam3"
fi
pip install -e "$HOME/src/sam3"

# everything else (laxnet + the nllcv pipeline it builds on, RF-DETR for the ball detector)
HERE="$(cd "$(dirname "$0")/.." && pwd)"
pip install -r "$HERE/requirements-gpu.txt"

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
import sam3, transformers, umap, anthropic
print("sam3, transformers, umap, anthropic: ok")
import rfdetr
print("rfdetr", getattr(rfdetr, "__version__", "?"), "ok; torch still", torch.__version__)
EOF
echo "Done. Next: hf auth login (after access to facebook/sam3 is granted), then sbatch slurm/laxnet_gpu.slurm"
echo "If pip replaced torch while installing rfdetr, rerun the torch cu128 line above."
