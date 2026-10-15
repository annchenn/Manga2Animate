# FLUX.2 storyboard backend

Draws each storyboard panel with [FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
instead of Gemini, then tiles the chosen panels into the same 3×2 sheet the Gemini
backend returns as a single image.

```bash
bash opensource_backends/flux2/setup_env.sh      # once: create the sb_diffusers env
bash opensource_backends/flux2/verify_setup.sh   # check env + conda + weights + GPU
python src/run_all.py -t <case> --storyboard-backend flux2
```

Per panel: generate `FLUX2_NUM_VARIANTS` variants conditioned on the source manga
panel **and** that panel's character reference crops → a Gemini reviewer scores
each variant → the winners are composed into `storyboard_0.png`. The flux2 path
produces exactly one storyboard candidate, where the Gemini backend produces
`NUM_STORYBOARDS` independent whole-sheet candidates.

## Layout

| file | role |
|---|---|
| `bridge.py` | builds the prompts and job spec; picks the GPU; launches `worker.py` in the `sb_diffusers` env |
| `worker.py` | the only file importing torch/diffusers; runs in that env and must never import `src.*` |
| `evaluate_panel_candidates.py` | Gemini best-of-N reviewer, writes `selected/panel{id}.png` |
| `grid_compose.py` | composes the final sheet |
| `prompts/panel_candidate_reviewer.txt` | the reviewer's prompt (flux2-only, so it ships here) |
| `run_local_smoke.py` | FLUX-only end-to-end check, **zero** Gemini calls |
| `verify_setup.sh` | pre-flight: conda, env, weights, GPU |
| `setup_env.sh` | creates the `sb_diffusers` conda env |

Two conda envs are mandatory, not a preference: MAGI (`src/magi.py`) pins
`transformers==4.38.0` while `Flux2Pipeline` needs 5.x. That conflict is the whole
reason `bridge.py` → `worker.py` is a subprocess instead of an import.

## Execution modes

`FLUX2_OFFLOAD_MODE` picks how the ~113GB of bf16 weights are placed.

| mode | needs free VRAM | per-call weight movement | when |
|---|---|---|---|
| `resident` (default) | ~75GB | none | a 97GB-class card |
| `cpu_offload` | ~60GB | streams ~113GB **every** `pipe()` call | smaller cards |

`resident` works because of two facts about `Flux2Pipeline`: `encode_prompt()`
short-circuits entirely when handed `prompt_embeds`, and reference images are
encoded by the **VAE**, never the text encoder. So the 48GB Mistral text encoder is
used once for all panel prompts, then released, leaving transformer + VAE (~64.8GB)
resident for the whole generation loop.

Releasing it means `pipe.text_encoder = None`, not `.to("cpu")`. Both
`DiffusionPipeline.device` and `._execution_device` return the first *registered*
module's device, and `__call__` builds its latents there — so a text encoder that is
merely moved to CPU makes them report `cpu` while the transformer is on the GPU,
which would raise an `Expected all tensors to be on the same device` error.

There is no "whole pipeline on one GPU" mode: 113GB does not fit on a 97GB card.
`resident` is what that would have been, done so it fits.

A panel's variants share one prompt, so they *could* be generated in a single
batched call — but on compute-saturated hardware this buys no wall-clock time
(each image still costs the same, whether batched or not) while using
meaningfully more peak VRAM. Default is `1` for that reason; raise it only if
your GPU/resolution combination turns out not to be compute-saturated. If a
batch does run out of memory, the worker automatically retries that panel one
variant at a time.

## Configuration

All read from `.env`; see `.env.example`.

| variable | default | meaning |
|---|---|---|
| `FLUX2_OFFLOAD_MODE` | `resident` | `resident` or `cpu_offload` (see above) |
| `FLUX2_CUDA_DEVICE` | `auto` | `CUDA_VISIBLE_DEVICES` for the worker; `auto` = least-loaded card |
| `FLUX2_BATCH_VARIANTS` | `1` | variants per `pipe()` call; measured to give no wall-clock benefit (see above), so left at 1 |
| `FLUX2_MIN_FREE_VRAM_GB` | mode default | abort before loading if the card has less free |
| `FLUX2_MAX_CONSECUTIVE_FAILURES` | `2` | abort the run after this many consecutive failures |
| `FLUX2_HF_OFFLINE` | `true` | skip re-validating the local weight cache against huggingface.co on every load (set `false` only for a first-time download) |
| `FLUX2_NUM_VARIANTS` | `3` | candidates generated per panel |
| `FLUX2_PANEL_WIDTH` / `_HEIGHT` | `768` / `640` | generated size **and** grid cell size |
| `FLUX2_NUM_INFERENCE_STEPS` | `28` | |
| `FLUX2_GUIDANCE_SCALE` | `4.0` | |
| `FLUX2_CONDA_ENV` | `sb_diffusers` | env the worker runs in |
| `FLUX2_CONDA_EXE` | auto-detected | absolute path to `conda`; needed when it is off PATH |
| `FLUX2_HF_HOME` | — | `HF_HOME` for the worker only (where the 105GB lives) |
| `FLUX2_HF_TOKEN_FILE` | `~/.cache/huggingface/token` | fallback when `HF_TOKEN` is unset |
| `FLUX2_WORKER_TIMEOUT` | `10800` | subprocess timeout, seconds |

`FLUX2_CPU_OFFLOAD` is deprecated but still honoured when `FLUX2_OFFLOAD_MODE` is unset.

## Weights

FLUX.2-dev is **gated** — the account behind your token must accept its licence
first. Only the component folders are needed (~105GB); the repo's root
`flux2-dev.safetensors` and `ae.safetensors` are duplicate single-file formats
`Flux2Pipeline` never reads, so excluding them saves ~65GB:

```bash
HF_HOME=/path/to/cache hf download black-forest-labs/FLUX.2-dev \
  --exclude "flux2-dev.safetensors" --exclude "ae.safetensors"
```

Point `FLUX2_HF_HOME` at that cache. `verify_setup.sh` confirms the snapshot
resolves fully offline — worth doing, since a single missing 155-byte config file
is enough to stop `from_pretrained`.

## Loading time, and why the weight cache should be on local disk

**Put `FLUX2_HF_HOME` on a local disk, not NFS.** `from_pretrained()`'s own
"loaded in Ns" progress bar is misleading — with `low_cpu_mem_usage` it only builds
memory-map handles; it does not read the actual bytes. The real disk read is
deferred to the moment something touches each tensor (e.g. `pipe.text_encoder.to
("cuda")`). Over a slow network mount this turns into a much slower read than
local disk, adding minutes to every single run, not just the first — moving the
cache to local disk fixes it permanently.

If a run instead sits at `Fetching N files: 0%` for minutes with no disk activity
at all, that's a different symptom: `from_pretrained()` re-validating the Hub's
metadata over the network instead of trusting the local cache.
`FLUX2_HF_OFFLINE=true` (the default) skips that round-trip entirely.

Loading the text encoder (used once, then released) takes on the order of a
minute or two; generating each image takes roughly a minute at the default
settings. Both scale with your hardware and the weight cache's storage speed, so
treat these as a rough starting point, not a guarantee.

## When it fails

**By far the most common cause is another process on the GPU.** FLUX.2-dev needs
most of a card, so a co-tenant holding tens of GB makes it OOM mid-run — and once
that happens, subsequent calls can fail with confusing device-mismatch errors
rather than a clear out-of-memory message.

Three things guard against this: the worker checks free VRAM *before* loading, it
aborts after `FLUX2_MAX_CONSECUTIVE_FAILURES` instead of grinding through
guaranteed-to-fail calls, and `run_local_smoke.py` refuses to start unless a card
is genuinely idle. Check `nvidia-smi`, pin an idle card with
`FLUX2_CUDA_DEVICE=<index>`, and rerun — finished images are kept and skipped
(`candidates/flux2_candidates_meta.json` is rewritten after every batch).

## Checking it works

```bash
bash opensource_backends/flux2/verify_setup.sh   # zero GPU time: conda, env, weights, GPU idleness

# real generation, zero API calls (Gemini reviewer skipped, a fixed variant picked instead)
python opensource_backends/flux2/run_local_smoke.py --reference <run dir> --target-dir <target dir> \
    --panels 1 --variants 1
python opensource_backends/flux2/run_local_smoke.py --reference <run dir> --target-dir <target dir> \
    --panels 1 --variants 3   # does batching fit?
python opensource_backends/flux2/run_local_smoke.py --reference <run dir> --target-dir <target dir>
    # full sheet
```

`<run dir>` is any completed `results/<target>/<run>` (needs `iter<N>/bestScript.json` +
`character/character_description.json`); `<target-dir>` is the `target/<target>` it used.

## Metrics

Every run writes `metrics.json` beside `config.json` (and appends a row to
`results/metrics_log.jsonl` so examples can be compared), containing per-stage wall
clock, the flux2 worker's own timings and peak VRAM, and the Gemini call count.

Two call counts are reported because they differ and only one is money:
`api_calls` is requests actually sent **including retries** (`with_retry` retries up
to 20×), `logical_calls` is pipeline steps. `failures` is tracked separately because
`with_retry` returns its default instead of raising, so a run can look clean while
containing calls that never succeeded.

Mind which number you are quoting: `run_local_smoke.py` skips the Gemini reviewer,
so its wall clock covers **generation + composition only** and its `api_calls` is 0.
A full-pipeline figure needs `run_all.py`. `metrics.json` records
`storyboard_backend` and which stages ran (`stages_skipped` lists what auto-resume
skipped — deliberately not recorded as 0 seconds, which would flatter a resumed run).
