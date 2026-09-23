#!/bin/bash
# Serve MiniMax-H3 (Ref2VA) locally with SGLang, for `create_video.py -m minimax-local`.
# Leave it running in its own shell; the pipeline talks to it over HTTP.
#
# Usage:
#   bash opensource_backends/minimax_h3/serve.sh                 # GPUs 0,1, port 30011
#   GPUS=0 bash opensource_backends/minimax_h3/serve.sh          # single GPU (offload, slower)
#
# Env vars:
#   GPUS        comma-separated CUDA device indices (default 0,1)
#   PORT        serve port (default 30011; must match MINIMAX_LOCAL_PORT in .env)
#   MODEL_DIR   weights directory (default $OSVG_MODELS_DIR/MiniMax-H3)
#   CONDA_ENV   conda env with sglang[diffusion] (default minimax_h3)
#   CONDA_BASE  conda install dir, only needed when conda is not on PATH
#   MODE        --performance-mode: memory | speed | auto (default memory)
#
# H3 pins keep_resident_min_available_gb=120 for (dit, text_encoder, vae), so
# "speed" (all-resident) OOMs while loading the transformer on 95GB cards;
# "memory" turns on layerwise/component offload instead.

set -e

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# OSVG_MODELS_DIR comes from the environment or .env, relative to the repo root.
if [ -z "$OSVG_MODELS_DIR" ] && [ -f "$REPO_ROOT/.env" ]; then
    OSVG_MODELS_DIR=$(grep -m1 '^OSVG_MODELS_DIR=' "$REPO_ROOT/.env" | cut -d= -f2-)
fi
OSVG_MODELS_DIR=${OSVG_MODELS_DIR:-opensource_video_generator/models}
case "$OSVG_MODELS_DIR" in
    /*) ;;
    *) OSVG_MODELS_DIR="$REPO_ROOT/$OSVG_MODELS_DIR" ;;
esac

GPUS=${GPUS:-0,1}
PORT=${PORT:-30011}
MODEL_DIR=${MODEL_DIR:-$OSVG_MODELS_DIR/MiniMax-H3}
CONDA_ENV=${CONDA_ENV:-minimax_h3}
MODE=${MODE:-memory}
# performance_mode=memory only offloads the DEFAULT group (text_encoder,
# image_encoder, vae); the DiT stays resident, which OOMs at 15s/360 frames on
# a 95GB card. "dit default" adds the DiT group; its async H2D prefetch hides
# most of the streaming cost.
LAYERWISE=${LAYERWISE:-dit default}
PREFETCH=${PREFETCH:-2}

if [ ! -f "$MODEL_DIR/model_index.json" ]; then
    echo "MiniMax-H3 weights not found in $MODEL_DIR (see opensource_backends/README.md)" >&2
    exit 1
fi

# A bare non-login shell (nohup/setsid) may not have conda on PATH; set
# CONDA_BASE (e.g. ~/miniconda3) in that case.
CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}
if [ -z "$CONDA_BASE" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "conda not found: put it on PATH or set CONDA_BASE=<conda install dir>" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

NGPU=$(awk -F, '{print NF}' <<< "$GPUS")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "serving ref2va from $MODEL_DIR on GPUs $GPUS (n=$NGPU), port $PORT"
echo "  mode=$MODE  layerwise=[$LAYERWISE]  prefetch=$PREFETCH"

CUDA_VISIBLE_DEVICES=$GPUS conda run --no-capture-output -n "$CONDA_ENV" \
    sglang serve \
    --model-path "$MODEL_DIR" \
    --model-variant ref2va \
    --num-gpus "$NGPU" \
    --ulysses-degree "$NGPU" \
    --performance-mode "$MODE" \
    --layerwise-offload-components $LAYERWISE \
    --dit-offload-prefetch-size "$PREFETCH" \
    --host 127.0.0.1 \
    --port "$PORT"
