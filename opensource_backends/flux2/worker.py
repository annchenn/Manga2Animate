"""
Runs INSIDE the `sb_diffusers` conda env (never imported by Manga2Animate's own
env, and deliberately free of any `src.*` / sibling-module import so it stays
loadable there). Reads a job-spec JSON written by bridge.py, generates
panels x variants with FLUX.2-dev, writes each image as soon as it's ready plus a
resumable manifest (written after every batch; already-finished images are skipped
on rerun).

Usage (only ever invoked via `conda run -n sb_diffusers`):
    python opensource_backends/flux2/worker.py --job <path/to/flux2_job.json>

Two execution modes, chosen by the job spec's "offload_mode":

  resident (default) -- the fast path, and why it is possible:
      diffusers' Flux2Pipeline.encode_prompt() short-circuits completely when it
      is handed `prompt_embeds` (pipeline_flux2.py: `if prompt_embeds is None:`),
      and FLUX.2 pushes its reference images through the *VAE*, never through the
      text encoder. So the 48GB Mistral text encoder is needed exactly once, for a
      handful of short prompts, and can then be dropped -- leaving transformer
      (~64.4GB) + vae (~0.3GB) resident on a single card for the whole generation
      loop with no weight movement at all.

  cpu_offload -- the older, much slower fallback for cards that cannot hold ~65GB.
      enable_model_cpu_offload() keeps ~113GB of weights in CPU RAM and streams
      submodules to the GPU on EVERY pipe() call, so a 6-panel x 3-variant job
      moves on the order of 2TB over PCIe. Kept because it is the path that was
      validated end-to-end on a 80GB-class GPU, but it is not the default here.

Note there is no `pipe.to("cuda")` mode: FLUX.2-dev is ~113GB in bf16, so moving
the *whole* pipeline onto one 97GB card simply OOMs. "resident" is what that mode
was reaching for, done in a way that actually fits.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import os

# Rough resident-set sizes of FLUX.2-dev in bf16, used only for the pre-flight
# free-VRAM check's default threshold (transformer 64.4 + vae 0.3, plus headroom
# for activations; cpu_offload's peak was measured at ~55GB on a real run).
_DEFAULT_MIN_FREE_GB = {"resident": 75.0, "cpu_offload": 60.0}


def _variant_path(out_dir: Path, pid, v: int) -> Path:
    return out_dir / f"panel{pid}_v{v}.png"


def _missing_variants(manifest_entry: dict, out_dir: Path, pid, num_variants: int) -> list[int]:
    """Variant numbers still to generate for one panel.

    A variant counts as done only when the manifest says "ok" AND the file is
    actually on disk -- the two can disagree if a previous run was killed between
    saving an image and rewriting the manifest, or if someone deleted an image to
    force a regen.
    """
    ok = {e["v"] for e in manifest_entry.get("variants", []) if e.get("status") == "ok"}
    return [v for v in range(1, num_variants + 1)
            if v not in ok or not _variant_path(out_dir, pid, v).exists()]


def _chunks(items: list[int], size: int) -> list[list[int]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _preflight_vram(torch, mode: str, required_gb: float) -> None:
    """Fail before loading 100+GB of weights if the chosen GPU is already busy.

    The one local run of this backend died the other way round: it loaded fine,
    produced panel 1, then OOMed on panel 2 because foreign processes held ~40GB
    of the card -- after which enable_model_cpu_offload's half-moved hooks turned
    every remaining call into a confusing device-mismatch error. Checking up front
    converts that into one clear message naming the actual problem.
    """
    if not torch.cuda.is_available():
        raise SystemExit("[flux2_worker] FATAL: no CUDA device visible to this process.")
    free_b, total_b = torch.cuda.mem_get_info()
    free_gb, total_gb = free_b / 2**30, total_b / 2**30
    name = torch.cuda.get_device_name(0)
    print(f"[flux2_worker] GPU: {name} | {free_gb:.1f}GB free of {total_gb:.1f}GB "
          f"| mode={mode} needs ~{required_gb:.0f}GB", flush=True)
    if free_gb < required_gb:
        raise SystemExit(
            f"[flux2_worker] FATAL: only {free_gb:.1f}GB free on {name} but mode "
            f"'{mode}' needs ~{required_gb:.0f}GB. Another process is most likely "
            f"holding memory on this card ({total_gb - free_gb:.1f}GB is in use). "
            f"Pick an idle GPU with FLUX2_CUDA_DEVICE=<index>, lower the requirement "
            f"with FLUX2_MIN_FREE_VRAM_GB, or wait for the other process to finish."
        )


# Exactly the files Flux2Pipeline.from_pretrained() reads (confirmed against its
# __init__ signature: vae, text_encoder, tokenizer, scheduler, transformer). The
# repo's two root files (flux2-dev.safetensors, ae.safetensors, ~65GB combined) are
# duplicate/alternate-format checkpoints it never touches, so a cache missing only
# those (plus a few cosmetic repo-root files: README/LICENSE/teaser images) is
# functionally complete even though it is not byte-for-byte "the whole repo".
_PIPELINE_ALLOW_PATTERNS = [
    "model_index.json", "transformer/*", "text_encoder/*",
    "vae/*", "tokenizer/*", "scheduler/*",
]


def _resolve_model_source(model_id: str) -> str:
    """repo id (online mode) or a local snapshot directory path (offline mode).

    diffusers' own from_pretrained(repo_id) cannot do this itself when offline:
    its download() only computes a *precise* allow_patterns list when it can reach
    the Hub for model_info() (to learn which files belong to which component). When
    that call is blocked -- which HF_HUB_OFFLINE is supposed to cause on purpose --
    it falls through to asking snapshot_download() to verify the ENTIRE repo is
    cached with no patterns at all, which fails here because our cache deliberately
    skips the two duplicate root checkpoints above. Confirmed via a real run: a
    7+ minute hang talking to huggingface.co with HF_HUB_OFFLINE unset, then with
    it set, an immediate "model is not cached locally" error despite every file the
    pipeline actually reads being present.

    The fix: resolve the snapshot folder ourselves with the SAME narrow patterns,
    and hand from_pretrained() that directory instead of the repo id --
    from_pretrained() skips download() entirely for a local directory
    (os.path.isdir() check), so none of the above ever runs."""
    if os.environ.get("HF_HUB_OFFLINE", "").strip() not in {"1", "true", "yes"}:
        return model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, local_files_only=True, allow_patterns=_PIPELINE_ALLOW_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="Path to job-spec JSON written by bridge.py")
    args = ap.parse_args()

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))

    import torch
    from diffusers import Flux2Pipeline
    from PIL import Image

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(job["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    mode = job.get("offload_mode", "resident")
    if mode not in _DEFAULT_MIN_FREE_GB:
        raise SystemExit(f"[flux2_worker] FATAL: unknown offload_mode {mode!r} "
                         f"(expected one of {sorted(_DEFAULT_MIN_FREE_GB)}).")
    num_variants = job["num_variants"]
    batch_size = max(1, min(job.get("batch_variants", 1), num_variants))
    max_consecutive = job.get("max_consecutive_failures", 2)
    width, height = job["panel_width"], job["panel_height"]

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Work out what is actually missing BEFORE loading anything, so a fully
    # resumed job costs nothing and a partially resumed one only encodes the
    # prompts it still needs.
    todo: dict[str, list[int]] = {}
    for panel in job["panels"]:
        key = str(panel["id"])
        entry = manifest.setdefault(
            key, {"prompt": panel["prompt"], "crop_ids": panel["crop_ids"], "variants": []}
        )
        entry["prompt"] = panel["prompt"]          # keep in sync if the script changed
        entry["crop_ids"] = panel["crop_ids"]
        missing = _missing_variants(entry, out_dir, panel["id"], num_variants)
        if missing:
            todo[key] = missing
        else:
            print(f"[flux2_worker] panel {panel['id']}: all {num_variants} variant(s) "
                  f"already done — skipping", flush=True)

    if not todo:
        print("[flux2_worker] nothing to do — every requested image already exists", flush=True)
        save_manifest()
        return

    required_gb = job.get("min_free_vram_gb") or _DEFAULT_MIN_FREE_GB[mode]
    _preflight_vram(torch, mode, required_gb)

    model_source = _resolve_model_source(job["model_id"])
    offline = model_source != job["model_id"]
    print(f"[flux2_worker] loading {job['model_id']} (bf16, {mode}"
          f"{', from local cache' if offline else ''}) ...", flush=True)
    t0 = time.time()
    pipe = Flux2Pipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    model_load_s = time.time() - t0
    print(f"[flux2_worker] weights loaded in {model_load_s:.1f}s", flush=True)

    # ---- Phase 1: prompts -> embeddings, then get the text encoder out of the way.
    prompt_embeds: dict[str, "torch.Tensor"] = {}
    text_encode_s = 0.0
    if mode == "resident":
        t0 = time.time()
        pipe.text_encoder.to("cuda")
        with torch.no_grad():
            for panel in job["panels"]:
                key = str(panel["id"])
                if key not in todo:
                    continue
                embeds, _text_ids = pipe.encode_prompt(
                    prompt=panel["prompt"], device="cuda", num_images_per_prompt=1,
                )
                # Park on CPU: these are tiny next to the weights, and holding them
                # on-device would just reduce the headroom available to attention.
                prompt_embeds[key] = embeds.to("cpu")

        # Setting the attribute to None is load-bearing, NOT tidiness: both
        # DiffusionPipeline.device and ._execution_device scan the registered
        # components and return the first torch.nn.Module's device. With the text
        # encoder merely moved back to CPU it is still registered, so they report
        # "cpu" and __call__ then builds its latents on the CPU while the
        # transformer sits on the GPU -- reproducing exactly the "Expected all
        # tensors to be on the same device" failure this backend hit before.
        # register_modules() treats None as a first-class value, and both
        # properties skip non-Module entries, so nulling it makes them resolve to
        # cuda. Verified directly against diffusers 0.39 (tests/test_flux2_resident.py).
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        text_encode_s = time.time() - t0
        print(f"[flux2_worker] encoded {len(prompt_embeds)} prompt(s) in "
              f"{text_encode_s:.1f}s; text encoder released", flush=True)

        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        print(f"[flux2_worker] transformer+vae resident on GPU "
              f"({torch.cuda.memory_allocated() / 2**30:.1f}GB allocated)", flush=True)
    else:
        pipe.enable_model_cpu_offload()

    # ---- Phase 2: generate.
    panel_timings: dict[str, float] = {}
    consecutive_failures = 0
    generation_start = time.time()

    def record(pid, v: int, seed, status: str):
        key = str(pid)
        entry = manifest[key]
        entry["variants"] = [e for e in entry["variants"] if e["v"] != v]
        entry["variants"].append(
            {"v": v, "seed": seed, "file": _variant_path(out_dir, pid, v).name, "status": status}
        )
        entry["variants"].sort(key=lambda e: e["v"])

    def generate(panel, variants: list[int], images) -> None:
        """One pipe() call covering `variants` of this panel; saves and records each."""
        key = str(panel["id"])
        generators, seeds = [], []
        for _v in variants:
            g = torch.Generator(device="cpu")
            seeds.append(g.seed())   # .seed() picks a fresh random seed AND returns it,
            generators.append(g)     # so the manifest value really does reproduce the image
        kwargs = dict(
            image=images, height=height, width=width,
            num_inference_steps=job["num_inference_steps"],
            guidance_scale=job["guidance_scale"],
            generator=generators,
            num_images_per_prompt=len(variants),
        )
        # Variants of a panel share one prompt, so encode_prompt() repeats the
        # batch-1 embedding up to num_images_per_prompt internally.
        if mode == "resident":
            kwargs["prompt_embeds"] = prompt_embeds[key].to("cuda")
        else:
            kwargs["prompt"] = panel["prompt"]

        out = pipe(**kwargs)
        if len(out.images) != len(variants):
            raise RuntimeError(
                f"asked for {len(variants)} image(s) but the pipeline returned "
                f"{len(out.images)} — refusing to guess which variant is which"
            )
        for i, v in enumerate(variants):
            path = _variant_path(out_dir, panel["id"], v)
            out.images[i].save(path)
            record(panel["id"], v, seeds[i], "ok")
            print(f"[flux2_worker] panel {panel['id']} v{v}: saved {path.name}", flush=True)
        del out

    for panel in job["panels"]:
        key = str(panel["id"])
        if key not in todo:
            continue

        panel_img = Image.open(panel["panel_image"]).convert("RGB")
        ref_imgs = [Image.open(p).convert("RGB") for p in panel["reference_images"]]
        images = [panel_img, *ref_imgs]

        panel_start = time.time()
        for batch in _chunks(todo[key], batch_size):
            try:
                generate(panel, batch, images)
                consecutive_failures = 0
            except torch.OutOfMemoryError as e:
                # Batching several variants into one call is the main memory
                # unknown here (reference images push the conditioning sequence to
                # ~14k tokens), so degrade to one-at-a-time before giving up.
                torch.cuda.empty_cache()
                gc.collect()
                if len(batch) > 1:
                    print(f"[flux2_worker] panel {panel['id']}: OOM on a batch of "
                          f"{len(batch)} — retrying one variant at a time", file=sys.stderr, flush=True)
                    for v in batch:
                        try:
                            generate(panel, [v], images)
                            consecutive_failures = 0
                        except torch.OutOfMemoryError as e2:
                            consecutive_failures += 1
                            record(panel["id"], v, None, f"error: {e2}")
                            print(f"[flux2_worker] panel {panel['id']} v{v}: OOM even "
                                  f"unbatched — {e2}", file=sys.stderr, flush=True)
                        finally:
                            torch.cuda.empty_cache()
                            gc.collect()
                        save_manifest()
                else:
                    consecutive_failures += 1
                    for v in batch:
                        record(panel["id"], v, None, f"error: {e}")
                    print(f"[flux2_worker] panel {panel['id']}: OOM — {e}",
                          file=sys.stderr, flush=True)
            except Exception as e:
                consecutive_failures += 1
                for v in batch:
                    record(panel["id"], v, None, f"error: {e}")
                print(f"[flux2_worker] panel {panel['id']} variants {batch}: FAILED — {e}",
                      file=sys.stderr, flush=True)
            finally:
                if mode == "cpu_offload":
                    torch.cuda.empty_cache()
                gc.collect()

            save_manifest()

            # Fail fast. Previously a single OOM was followed by 15 doomed calls
            # (each failing with a misleading device-mismatch error) before the
            # bridge finally reported "15 candidate images are still missing" --
            # burning GPU time and hiding the real cause. Stop at the source.
            if consecutive_failures >= max_consecutive:
                save_manifest()
                raise SystemExit(
                    f"[flux2_worker] FATAL: aborting after {consecutive_failures} "
                    f"consecutive failures. The usual cause is insufficient free VRAM "
                    f"(another process sharing this GPU). Check `nvidia-smi`, then pin an "
                    f"idle card with FLUX2_CUDA_DEVICE=<index>. Images already written are "
                    f"kept and will be skipped on the next run."
                )

        panel_timings[key] = time.time() - panel_start
        print(f"[flux2_worker] panel {panel['id']}: {panel_timings[key]:.1f}s "
              f"for {len(todo[key])} variant(s)", flush=True)

    # ---- Metrics for bridge.py to fold into the run's metrics.json. Written into
    # the manifest rather than passed back over the pipe, so nothing has to cross
    # the conda-env boundary.
    images_made = sum(
        1 for k, e in manifest.items()
        if not k.startswith("_") and isinstance(e, dict)
        for v in e.get("variants", []) if v.get("status") == "ok"
    )
    generated_now = sum(len(v) for v in todo.values())
    manifest["_metrics"] = {
        "offload_mode": mode,
        "batch_variants": batch_size,
        "model_load_s": round(model_load_s, 2),
        "text_encode_s": round(text_encode_s, 2),
        "generation_s": round(time.time() - generation_start, 2),
        "per_panel_s": {k: round(v, 2) for k, v in panel_timings.items()},
        "images_generated_this_run": generated_now,
        "images_total_ok": images_made,
        "per_image_s_mean": (
            round((time.time() - generation_start) / generated_now, 2) if generated_now else None
        ),
        "peak_vram_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    save_manifest()

    print(f"[flux2_worker] done: {generated_now} image(s) generated this run "
          f"({images_made} ok in total) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
