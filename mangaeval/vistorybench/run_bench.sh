#!/usr/bin/env bash
# Run ViStoryBench on our dataset.
#
# Converts any non-jpg character reference images first (CIDS only reads .jpg),
# then runs the evaluation. All arguments are passed through to bench_run.py.
#
#   ./run_bench.sh --method mybaseline --mode base
#   ./run_bench.sh --method mybaseline --mode base --metrics cids csd
#   GPU=3 ./run_bench.sh --method mybaseline --mode base
#
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-$(command -v python)}"
GPU="${GPU:-0}"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: python not found at $PYTHON" >&2
    echo "       set PYTHON=/path/to/env/bin/python, or activate the vistorybench env" >&2
    exit 1
fi

if [ ! -d data/pretrain ]; then
    echo "ERROR: data/pretrain is missing. Run: sh download_weights.sh" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "usage: $0 --method <name> --mode <name> [--metrics ...] [other bench_run.py args]" >&2
    exit 1
fi

echo "==> Normalizing character reference images to jpg"
"$PYTHON" src/normalize_ref_images.py

echo
echo "==> Running evaluation on GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" bench_run.py --language en "$@"

echo
echo "==> Writing result.md"
"$PYTHON" src/make_result_md.py
