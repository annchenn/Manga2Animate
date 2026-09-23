#!/usr/bin/env bash
# Creates (or reuses) a standalone conda env for this aesthetic-scoring tool.
# Kept SEPARATE from every other env in this repo (main pipeline, flux2, VBench,
# vistorybench all pin their own torch/transformers versions) -- this tool is a
# CLI, not something any of them import, so there is no reason to share an env.
#
# Usage:
#   bash setup_env.sh
#   AESTHETIC_CONDA_ENV=myenv AESTHETIC_TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash setup_env.sh
set -euo pipefail

ENV_NAME="${AESTHETIC_CONDA_ENV:-manga2animate-aesthetic}"
TORCH_INDEX="${AESTHETIC_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "Creating conda env '${ENV_NAME}' (python 3.11)..."
    conda create -n "${ENV_NAME}" python=3.11 -y
else
    echo "conda env '${ENV_NAME}' already exists — skipping creation."
fi

conda run -n "${ENV_NAME}" pip install --upgrade pip
# torch/torchvision first, pinned to a CUDA build, so nothing later pulls a
# CPU-only or mismatched wheel. Override AESTHETIC_TORCH_INDEX for a different
# CUDA version, or set it to an empty string for a CPU-only install.
if [ -n "${TORCH_INDEX}" ]; then
    conda run -n "${ENV_NAME}" pip install torch torchvision --index-url "${TORCH_INDEX}"
else
    conda run -n "${ENV_NAME}" pip install torch torchvision
fi
conda run -n "${ENV_NAME}" pip install -r "${HERE}/requirements.txt"

echo
echo "Verifying..."
conda run -n "${ENV_NAME}" python - <<'PYCODE'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import clip
print("clip ok:", clip.available_models())
PYCODE

echo
echo "Done. Run scores with, e.g.:"
echo "  conda run -n ${ENV_NAME} python ${HERE}/score.py <path> [<path> ...]"
echo "See README.md for the HF_HOME cache-reuse tip and the aesthetic_shadow_v2 license note."
