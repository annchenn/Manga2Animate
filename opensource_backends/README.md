# Open-source backends

The pipeline can run fully on open models: FLUX.2-dev for the storyboard and
MiniMax-H3 or Index-AniSora for the video. Each needs its own conda env, so this
folder holds both the out-of-band setup and — for flux2 — the backend code itself;
`src/` keeps only a thin dispatch into it.

| stage | selected with | model | how the pipeline reaches it |
|---|---|---|---|
| storyboard | `run_all.py --storyboard-backend flux2` | FLUX.2-dev | `flux2/bridge.py` runs `flux2/worker.py` in the `sb_diffusers` env ([details](flux2/README.md)) |
| video | `create_video.py -m minimax-local` | MiniMax-H3 (Ref2VA), reference-conditioned | HTTP to a running SGLang server (`minimax_h3/serve.sh`) |
| video | `create_video.py -m anisora` | Index-AniSora V3.2, first-frame (per panel) | runs upstream `generate_txt_new.py` in its own conda env |

## FLUX.2-dev storyboard (`--storyboard-backend flux2`)

```bash
bash opensource_backends/flux2/setup_env.sh      # once
bash opensource_backends/flux2/verify_setup.sh   # env + conda + weights + GPU
python src/run_all.py -t <target> --storyboard-backend flux2
```

`setup_env.sh` creates the `sb_diffusers` env with pinned versions (diffusers
0.39, torch 2.13, transformers 5.15), kept apart from the main env because MAGI
pins `transformers==4.38.0` — that conflict is why the backend talks to its model
through a subprocess rather than an import.

FLUX.2-dev is a gated HuggingFace repo: accept its license first and provide a
token (`HF_TOKEN`, or a token file at `FLUX2_HF_TOKEN_FILE`). Only the component
folders are needed (~105GB); excluding the repo's two root single-file checkpoints
saves ~65GB. Point `FLUX2_HF_HOME` at the cache.

It needs most of a 97GB-class GPU, and a co-tenant process on that card is the
most common cause of failure — set `FLUX2_CUDA_DEVICE` to an idle one.

**[`flux2/README.md`](flux2/README.md)** has the execution modes
(`resident` vs `cpu_offload`), the full `FLUX2_*` table, the metrics written per
run, and troubleshooting.

## Video model weights

Both video models expect their weights under one root, `OSVG_MODELS_DIR` in `.env`
(default `opensource_video_generator/models`, relative to the repo root):

```
$OSVG_MODELS_DIR/
├── MiniMax-H3/
└── Index-anisora/
```

A single model can be pointed elsewhere with `MINIMAX_LOCAL_WEIGHTS` or
`ANISORA_WEIGHTS`.

## MiniMax-H3 (`-m minimax-local`)

1. Environment (tested with sglang 0.5.18):

   ```bash
   conda create -n minimax_h3 python=3.12 -y
   conda run -n minimax_h3 pip install "sglang[diffusion]"
   ```

2. Weights. Only the `Ref2VA` variant is needed (~144GB of the 498GB repo),
   since the storyboard goes in as a reference image:

   ```bash
   hf download MiniMaxAI/MiniMax-H3 \
     --include "model_index.json" "LICENSE" "README.md" "docs/*" "scripts/*" "Ref2VA/*" \
     --local-dir $OSVG_MODELS_DIR/MiniMax-H3
   ```

3. Start the server and leave it running:

   ```bash
   GPUS=0,1 bash opensource_backends/minimax_h3/serve.sh
   ```

4. In another shell:

   ```bash
   python src/create_video.py -t <target> -m minimax-local
   ```

`serve.sh` options: `GPUS`, `PORT` (must match `MINIMAX_LOCAL_PORT`, default
30011), `MODEL_DIR`, `CONDA_ENV`, `MODE`, and `CONDA_BASE` when conda is not
on PATH (e.g. under `nohup`/`setsid`). Output is 768P / 24fps / 15s. Peak
memory was 28.3GB per GPU on two 95GB cards with the default offload settings.

## Index-AniSora V3.2 (`-m anisora`)

1. Code, plus our one-line patch (casts the fp32 A14B experts to bf16 before
   FSDP sharding; without it 2×48GB GPUs OOM):

   ```bash
   git clone https://github.com/bilibili/Index-anisora.git $OSVG_MODELS_DIR/Index-anisora
   cd $OSVG_MODELS_DIR/Index-anisora
   git checkout 914fd69
   git apply <repo>/opensource_backends/anisora/fsdp.patch
   ```

2. Environment, following upstream `anisoraV3.2/README.md`:

   ```bash
   conda create -n anisora3.2 python=3.10 -y
   conda activate anisora3.2
   cd anisoraV3.2
   pip install -r req-fastvideo.txt
   pip install -r requirements.txt
   pip install -e .
   ```

3. Weights: download `V3.2/` from
   [huggingface.co/IndexTeam/Index-anisora](https://huggingface.co/IndexTeam/Index-anisora/tree/main/V3.2)
   into `anisoraV3.2/Index-anisora-V3.2/V3.2/` (it must contain
   `high_noise_model/`, `low_noise_model/`, `models_t5_umt5-xxl-enc-bf16.pth`,
   `Wan2.1_VAE.pth`).

4. Run:

   ```bash
   python src/create_video.py -t <target> -m anisora
   ```

No server: each candidate is one synchronous `conda run` over all panels.
`ANISORA_GPU` (default `0,1`) picks devices; two or more switch to torchrun +
FSDP + Ulysses, which 1280×720 × 81 frames needs on 48GB cards. Other knobs:
`ANISORA_FRAME_NUM` (default 81 = 5s @ 16fps, or `auto`), `ANISORA_SIZE`,
`ANISORA_STEPS`, `ANISORA_ENV`.
