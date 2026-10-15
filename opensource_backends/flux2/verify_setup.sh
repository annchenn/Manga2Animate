#!/usr/bin/env bash
# Pre-flight for the flux2 storyboard backend. Read-only, no GPU work, ~seconds.
#
# Checks the four things that actually go wrong on a fresh machine, in the order
# they bite: the worker env, the conda launcher, the weights, and whether a GPU is
# free enough to use. Run this before a long job rather than discovering a missing
# 155-byte config file an hour in.
#
#   bash opensource_backends/flux2/verify_setup.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# .env is the source of truth for the pipeline, so read it here too.
if [ -f .env ]; then
    set -a; . ./.env; set +a
    echo "config: .env"
else
    echo "config: .env MISSING (copy .env.example and fill it in)"
fi

ENV_NAME="${FLUX2_CONDA_ENV:-sb_diffusers}"
MODEL_ID="${FLUX2_MODEL_ID:-black-forest-labs/FLUX.2-dev}"
MODE="${FLUX2_OFFLOAD_MODE:-resident}"
fail=0
note() { printf '  %-7s %s\n' "$1" "$2"; }

# ---- 1. conda launcher -------------------------------------------------------
echo
echo "[1/4] conda"
CONDA_BIN="${FLUX2_CONDA_EXE:-}"
if [ -z "$CONDA_BIN" ]; then CONDA_BIN="$(command -v conda 2>/dev/null || true)"; fi
if [ -z "$CONDA_BIN" ] && [ -x "$HOME/miniconda3/bin/conda" ]; then CONDA_BIN="$HOME/miniconda3/bin/conda"; fi
if [ -x "$CONDA_BIN" ]; then
    note OK "$CONDA_BIN ($("$CONDA_BIN" --version 2>&1))"
else
    note FAIL "no conda found. Set FLUX2_CONDA_EXE in .env to its absolute path."
    note "" "(conda is often off PATH in non-interactive shells.)"
    fail=1
fi

# ---- 2. worker env -----------------------------------------------------------
echo
echo "[2/4] worker env '$ENV_NAME'"
ENV_PY=""
if [ -x "$CONDA_BIN" ]; then
    ENV_PREFIX="$("$CONDA_BIN" env list 2>/dev/null | awk -v n="$ENV_NAME" '$1==n {print $NF}')"
    [ -n "${ENV_PREFIX:-}" ] && [ -x "$ENV_PREFIX/bin/python" ] && ENV_PY="$ENV_PREFIX/bin/python"
fi
if [ -n "$ENV_PY" ]; then
    "$ENV_PY" - <<'PY'
import sys
try:
    import torch, diffusers
    from diffusers import Flux2Pipeline  # noqa: F401
except Exception as e:
    print(f"  FAIL    cannot import the stack: {type(e).__name__}: {e}"); sys.exit(1)
print(f"  OK      diffusers {diffusers.__version__}, torch {torch.__version__}")
print(f"  OK      Flux2Pipeline importable")
caps = torch.cuda.get_arch_list()
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0); tag = f"sm_{cc[0]}{cc[1]}"
    ok = tag in caps
    print(f"  {'OK     ' if ok else 'WARN   '} {torch.cuda.get_device_name(0)} is {tag}; "
          f"torch built for {tag in caps and 'it' or 'other arches only'}")
    sys.exit(0 if ok else 0)
else:
    print("  WARN    torch reports no CUDA device"); sys.exit(0)
PY
    [ $? -ne 0 ] && fail=1
else
    note FAIL "env '$ENV_NAME' not found. Run: bash opensource_backends/flux2/setup_env.sh"
    fail=1
fi

# ---- 3. weights --------------------------------------------------------------
echo
echo "[3/4] weights ($MODEL_ID)"
if [ -n "$ENV_PY" ]; then
    FLUX2_HF_HOME="${FLUX2_HF_HOME:-}" MODEL_ID="$MODEL_ID" "$ENV_PY" - <<'PY'
import json, os, sys
hf_home = os.environ.get("FLUX2_HF_HOME") or ""
if hf_home:
    os.environ["HF_HOME"] = hf_home
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import snapshot_download
repo = os.environ["MODEL_ID"]
# Only what Flux2Pipeline.from_pretrained reads. The repo's root
# flux2-dev.safetensors / ae.safetensors are duplicate single-file formats it never
# touches, so a full-repo completeness check would fail on files nothing needs.
try:
    path = snapshot_download(repo, local_files_only=True,
        allow_patterns=["model_index.json", "transformer/*", "text_encoder/*",
                        "vae/*", "tokenizer/*", "scheduler/*"])
except Exception as e:
    print(f"  FAIL    not fully cached under HF_HOME={hf_home or '(default)'}")
    print(f"          {type(e).__name__}: {str(e)[:300]}")
    sys.exit(1)
total = 0
for comp in ("transformer", "text_encoder", "vae"):
    d = os.path.join(path, comp)
    shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    total += sum(os.path.getsize(os.path.realpath(os.path.join(d, f))) for f in shards)
    idx = [f for f in os.listdir(d) if f.endswith("index.json")]
    if idx:
        want = set(json.load(open(os.path.join(d, idx[0])))["weight_map"].values())
        missing = sorted(want - set(shards))
        if missing:
            print(f"  FAIL    {comp}: index expects {len(want)} shard(s), missing {missing}")
            sys.exit(1)
print(f"  OK      complete offline, {total/2**30:.1f} GiB of components")
print(f"  OK      {path}")
PY
    [ $? -ne 0 ] && fail=1
else
    note SKIP "needs the worker env"
fi

# ---- 4. GPU availability -----------------------------------------------------
echo
echo "[4/4] GPU (mode '$MODE' needs ~$([ "$MODE" = resident ] && echo 75 || echo 60)GB free)"
if command -v nvidia-smi >/dev/null 2>&1; then
    need=$([ "$MODE" = resident ] && echo 75 || echo 60)
    idle=0
    while IFS=, read -r idx used total; do
        idx=$(echo "$idx" | tr -d ' '); used=$(echo "$used" | tr -d ' '); total=$(echo "$total" | tr -d ' ')
        free_gb=$(( (total - used) / 1024 ))
        if [ "$free_gb" -ge "$need" ] && [ "$used" -le 1024 ]; then
            note OK "GPU$idx: ${free_gb}GB free, ${used}MiB used — idle"
            idle=$((idle+1))
        else
            note "-" "GPU$idx: ${free_gb}GB free, ${used}MiB used by another process"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
    if [ "$idle" -eq 0 ]; then
        note WARN "no idle GPU right now. The smoke test will refuse to start;"
        note "" "wait for a card to free up (this is advisory, not a setup error)."
    fi
else
    note SKIP "nvidia-smi not available"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "Setup OK. Next: python opensource_backends/flux2/run_local_smoke.py --panels 1 --variants 1"
else
    echo "Setup INCOMPLETE — fix the FAIL items above."
fi
exit "$fail"
