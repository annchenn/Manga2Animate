"""
Runs INSIDE the `sb_diffusers` conda env (never imported by Manga2Animate's own
env). Reads a job-spec JSON written by src/flux2_bridge.py, generates
panels x variants with FLUX.2-dev, writes each image as soon as it's ready plus a
resumable manifest (write after every image; skip already-existing files on
rerun).

Usage (only ever invoked via `conda run -n sb_diffusers`):
    python src/flux2_worker.py --job <path/to/flux2_job.json>
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="Path to job-spec JSON written by flux2_bridge.py")
    args = ap.parse_args()

    job = json.loads(Path(args.job).read_text(encoding="utf-8"))

    import torch
    from diffusers import Flux2Pipeline
    from PIL import Image

    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(job["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    cpu_offload = job.get("cpu_offload", True)
    mode = "cpu offload" if cpu_offload else "full GPU (no offload)"
    print(f"[flux2_worker] loading {job['model_id']} (bf16, {mode}) ...", flush=True)
    pipe = Flux2Pipeline.from_pretrained(job["model_id"], torch_dtype=torch.bfloat16)
    if cpu_offload:
        # Safe default: keeps most weights on
        # CPU RAM, streams submodules to GPU on demand. Works on small GPUs but
        # pays real PCIe-transfer overhead on every single pipe() call.
        pipe.enable_model_cpu_offload()
    else:
        # Much faster when the whole model actually fits in VRAM (e.g. a 96GB+
        # card with headroom to spare) — no offload/transfer overhead at all
        # across the panel x variant loop below. Can OOM on smaller GPUs; if it
        # does, set FLUX2_CPU_OFFLOAD=true in .env and rerun.
        pipe.to("cuda")

    num_variants = job["num_variants"]
    width, height = job["panel_width"], job["panel_height"]

    for panel in job["panels"]:
        pid = panel["id"]
        key = str(pid)
        manifest.setdefault(key, {"prompt": panel["prompt"], "crop_ids": panel["crop_ids"], "variants": []})
        done_v = {e["v"] for e in manifest[key]["variants"] if e.get("status") == "ok"}

        panel_img = Image.open(panel["panel_image"]).convert("RGB")
        ref_imgs = [Image.open(p).convert("RGB") for p in panel["reference_images"]]
        images = [panel_img, *ref_imgs]

        for v in range(1, num_variants + 1):
            out_path = out_dir / f"panel{pid}_v{v}.png"
            if v in done_v and out_path.exists():
                print(f"[flux2_worker] panel {pid} v{v}: skip (already done)", flush=True)
                continue
            seed = None  # populated below; stays None if we fail before it's set
            try:
                # Random seed per call (not derived from v) so different variants
                # -- and the same variant slot across different panels -- don't
                # all reuse the same fixed small seed; .seed() both initializes
                # and returns the seed actually used, so it can be recorded below
                # for reference (not for exact reproducibility across runs).
                generator = torch.Generator(device="cpu")
                seed = generator.seed()
                out = pipe(
                    prompt=panel["prompt"],
                    image=images,
                    height=height,
                    width=width,
                    num_inference_steps=job["num_inference_steps"],
                    guidance_scale=job["guidance_scale"],
                    generator=generator,
                )
                out.images[0].save(out_path)
                status = "ok"
                print(f"[flux2_worker] panel {pid} v{v}: saved {out_path.name}", flush=True)
            except Exception as e:
                status = f"error: {e}"
                print(f"[flux2_worker] panel {pid} v{v}: FAILED — {e}", file=sys.stderr, flush=True)

            manifest[key]["variants"] = [e for e in manifest[key]["variants"] if e["v"] != v]
            manifest[key]["variants"].append({"v": v, "seed": seed, "file": out_path.name, "status": status})
            manifest[key]["variants"].sort(key=lambda e: e["v"])
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[flux2_worker] done: {len(job['panels'])} panel(s) x {num_variants} variant(s) -> {out_dir}")


if __name__ == "__main__":
    main()
