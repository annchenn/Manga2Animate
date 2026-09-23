# Aesthetic scoring (Aesthetic Shadow v2 + Waifu Scorer v3)

Standalone, offline, no-reference aesthetic scoring for anime/manga-style
images and video frames. Point it at any image/video file or directory and
get back a per-item score CSV.

This is a plain CLI tool, kept separate from `../VBench/` and
`../vistorybench/` on purpose: those two already have their own incompatible
conda envs and dependency pins, and neither has a shared metric-plugin
mechanism. When either framework needs a score from here, call this script
from within its own environment (e.g. via `subprocess`) and read back the
CSV — do not try to `import` this package's modules into another env, since
its dependencies (CLIP, safetensors) aren't guaranteed to coexist with either
framework's pins.

## Metrics

| name | model | output | higher is better |
|---|---|---|---|
| `aesthetic_shadow_v2` | ViT binary classifier {hq, lq} | P(hq), 0.0–1.0 | yes |
| `waifu_scorer_v3` | CLIP ViT-L/14 features + trained MLP head | 0.0–10.0 | yes |

Both are ported as-is from a separate scoring experiment (`../../../Aestheticexp/`
on the author's machine) so scores stay comparable with numbers already
computed there — see `metrics.py` for the exact inference logic.

### License note

`aesthetic_shadow_v2`'s weights (`NeoChen1024/aesthetic-shadow-v2-backup`, a
community mirror of the now-deleted `shadowlilac/aesthetic-shadow-v2`) are
licensed **CC-BY-NC-4.0 (non-commercial)**. Fine for research/paper use;
do not use scores from this metric to support a commercial product without
checking that license first.

## Setup

```bash
bash setup_env.sh
```

This creates a conda env (`manga2animate-aesthetic` by default, override with
`AESTHETIC_CONDA_ENV=...`), installs a CUDA build of torch (`cu124` by
default, override with `AESTHETIC_TORCH_INDEX=...`, or set it to an empty
string for CPU-only), and installs `requirements.txt`. Or do it manually:

```bash
conda create -n manga2animate-aesthetic python=3.11 -y
conda activate manga2animate-aesthetic
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### First run downloads weights

The first score you compute downloads:
- `aesthetic_shadow_v2`: ~4 GB, to the standard Hugging Face cache (`HF_HOME`,
  default `~/.cache/huggingface`).
- `waifu_scorer_v3`: ~1 GB CLIP ViT-L/14 backbone (OpenAI `clip` package's own
  cache, default `~/.cache/clip`) + ~11 MB MLP head (Hugging Face cache).

If you've already run the Aestheticexp experiment on this machine, its
weights are already downloaded — reuse them instead of re-downloading by
pointing `HF_HOME` at its cache before running:

```bash
export HF_HOME=/path/to/Aestheticexp/.hf_cache
```

(`~/.cache/clip` is a global, package-level default that both tools already
share automatically — no extra configuration needed for the CLIP backbone.)

## Usage

```bash
conda run -n manga2animate-aesthetic python score.py <path> [<path> ...] [options]
```

`<path>` can be an image file, a video file, or a directory containing
either (searched recursively by default). Examples:

```bash
# Manga2Animate's own storyboard output for one target/run
python score.py ../../results/DualJustice1/20260901_120000/iter1/storyboard_panels/

# a vistorybench outputs tree for one method
python score.py ../vistorybench/data/outputs/ours_scene/base/en/20260901_120000/

# a folder of generated video clips
python score.py ../VBench/evaluation_runs/ours/videos/ --frames-per-video 10
```

### Options

| flag | default | meaning |
|---|---|---|
| `--metrics` | both | comma-separated subset, e.g. `--metrics aesthetic_shadow_v2` |
| `--out` | `./scores_<timestamp>.csv` | output CSV path |
| `--recursive` / `--no-recursive` | recursive | whether directories are searched recursively |
| `--frames-per-video` | `8` | frames sampled per video, evenly spaced (no shot-cut detection — this is a generic scorer, not story/shot-aware) |
| `--device` | auto-detect | `cuda` or `cpu` |

### Output

A CSV with one row per (item, metric):

```
path,content_type,frame_index,metric,score
.../panel1.png,image,,aesthetic_shadow_v2,0.8421
.../panel1.png,image,,waifu_scorer_v3,6.72
.../clip.mp4,video_frame,0,aesthetic_shadow_v2,0.71
.../clip.mp4,video_frame,1,aesthetic_shadow_v2,0.69
...
```

`content_type` is `image` or `video_frame`; `frame_index` is only set for
video frames. A per-path, per-metric mean is also printed to the terminal
after the CSV is written.

## Not done here

This tool does not integrate into VBench's `tools/` workflow or
vistorybench's `bench_run.py --metrics` pipeline (e.g. vistorybench's
`EVALUATOR_REGISTRY`) — it is deliberately just the shared scoring code plus
a CLI. Wiring it into either framework's automated run is a separate task.
