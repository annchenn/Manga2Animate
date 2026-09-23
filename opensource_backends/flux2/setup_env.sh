#!/usr/bin/env bash
# Creates (or reuses) the `sb_diffusers` conda env used by the flux2 storyboard
# backend (src/flux2_worker.py). Kept SEPARATE from Manga2Animate's own env: it
# needs diffusers>=0.39 + a recent torch/transformers, which conflicts with this
# repo's own transformers==4.38.0 pin (required by src/magi.py's MAGI model). See
# src/flux2_bridge.py / src/flux2_worker.py for the conda-run subprocess bridge,
# which exists for the same two-incompatible-dependency-stacks reason.
#
# Usage:
#   bash opensource_backends/flux2/setup_env.sh
set -euo pipefail

ENV_NAME="${FLUX2_CONDA_ENV:-sb_diffusers}"
HF_TOKEN_FILE="${FLUX2_HF_TOKEN_FILE:-$HOME/.cache/huggingface/token}"

# Pinned to the exact versions validated end-to-end against Flux2Pipeline (its
# calling convention — e.g. the image=[panel, *refs] multi-image conditioning
# used in src/flux2_worker.py — isn't guaranteed stable across diffusers releases,
# so an unpinned install here is a real reproducibility risk on a fresh machine).
if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "Creating conda env '${ENV_NAME}' (python 3.11)..."
    conda create -n "${ENV_NAME}" python=3.11 -y
    conda run -n "${ENV_NAME}" pip install --upgrade pip
    conda run -n "${ENV_NAME}" pip install \
        diffusers==0.39.0 torch==2.13.0 torchvision==0.28.0 transformers==5.15.0 \
        accelerate==1.14.0 safetensors==0.8.0 sentencepiece==0.2.2 protobuf==7.35.1 \
        pillow==12.3.0
else
    echo "conda env '${ENV_NAME}' already exists — skipping creation."
fi

echo
echo "FLUX.2-dev is a gated HF repo. Make sure the account behind your HF token has"
echo "accepted its license at https://huggingface.co/black-forest-labs/FLUX.2-dev"
if [ -f "${HF_TOKEN_FILE}" ]; then
    echo "Found HF token file at ${HF_TOKEN_FILE} — src/flux2_bridge.py reads it"
    echo "automatically and passes it to the worker subprocess as HF_TOKEN."
else
    echo "No HF token file at ${HF_TOKEN_FILE}."
    echo "Set HF_TOKEN directly in .env instead, or place your token at that path."
fi

echo
echo "Done. src/flux2_bridge.py invokes this env itself via"
echo "  conda run -n ${ENV_NAME} python src/flux2_worker.py --job <job.json>"
echo "whenever STORYBOARD_BACKEND=flux2 — you do not normally need to 'conda activate' it."
