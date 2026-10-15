"""FLUX-only smoke test: does this machine actually produce a storyboard?

Runs the real generation path -- bridge.build_job_spec -> worker.py in the
`sb_diffusers` env -> grid_compose -- but skips the Gemini panel reviewer, so it
makes ZERO API calls. Instead of the reviewer's choice it takes a fixed variant
per panel (--pick), which is enough to prove the two things in question: FLUX.2
generates panels here, and the panels tile into a final storyboard sheet.

Point --reference at any completed run dir (one containing iter<N>/bestScript.json
and character/character_description.json) and --target-dir at the target it used
(manga_panels/ and characters/) -- e.g. a run already produced by src/run_all.py.

    # cheapest end-to-end check: one panel, one variant
    python opensource_backends/flux2/run_local_smoke.py \
        --reference results/<target>/<run> --target-dir target/<target> \
        --panels 1 --variants 1
    # does batching fit in VRAM?
    python opensource_backends/flux2/run_local_smoke.py \
        --reference results/<target>/<run> --target-dir target/<target> \
        --panels 1 --variants 3
    # full sheet
    python opensource_backends/flux2/run_local_smoke.py \
        --reference results/<target>/<run> --target-dir target/<target>

Writes metrics.json next to the output (timings, peak VRAM, Gemini calls = 0).
By default it refuses to start unless a GPU is essentially idle, since FLUX.2-dev
needs most of a 97GB card and sharing one is what broke this backend before;
--allow-busy-gpu overrides that.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# A standalone entry point (not run through src/run_all.py, which would otherwise
# load .env first) -- must load it explicitly or every FLUX2_*/credential setting
# in .env is silently ignored. Confirmed as a real bug: FLUX2_HF_HOME went unread,
# so the worker fell back to the wrong (near-empty) default HF cache.
load_dotenv()


def gpu_rows() -> list[tuple[int, int, int]]:
    """[(index, used_MiB, total_MiB)] from nvidia-smi, or [] if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        try:
            i, u, t = (int(p.strip()) for p in line.split(","))
            rows.append((i, u, t))
        except ValueError:
            continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True,
                    help="completed run dir providing bestScript.json + character_description.json "
                         "(e.g. results/<target>/<run>)")
    ap.add_argument("--iter", default="0", help="which iteration's bestScript.json to use")
    ap.add_argument("--target-dir", type=Path, required=True,
                    help="target dir with manga_panels/ and characters/ (e.g. target/<target>)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: <repo>/results/_flux2_smoke/<timestamp>")
    ap.add_argument("--panels", type=int, default=0, help="first N panels only (0 = all)")
    ap.add_argument("--variants", type=int, default=None,
                    help="variants per panel (default: FLUX2_NUM_VARIANTS)")
    ap.add_argument("--pick", type=int, default=1,
                    help="variant number used as the 'selected' panel (no Gemini reviewer)")
    ap.add_argument("--min-free-vram-gb", type=float, default=80.0,
                    help="a GPU needs at least this much free to count as usable")
    ap.add_argument("--max-used-mib", type=int, default=1024,
                    help="and at most this much already in use, so it is genuinely idle "
                         "rather than merely roomy (0 other processes is the intent)")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="skip the idle-GPU gate")
    args = ap.parse_args()

    for p, what in ((args.reference, "reference run"), (args.target_dir, "target dir")):
        if not p.is_dir():
            print(f"ERROR: {what} not found: {p}", file=sys.stderr)
            return 2

    # ---- idle-GPU gate -----------------------------------------------------
    rows = gpu_rows()
    if rows:
        print("GPUs: " + ", ".join(
            f"GPU{i}={u}MiB used/{t}MiB ({(t-u)/1024:.0f}GB free)" for i, u, t in rows))
        # Both conditions: enough room AND nothing meaningful already on it. Free
        # memory alone is not enough -- a card with 84GB free still has another
        # process on it that can grow at any moment, and FLUX.2-dev OOMing mid-run
        # is precisely how this backend failed before.
        idle = [(i, u, t) for i, u, t in rows
                if (t - u) / 1024 >= args.min_free_vram_gb and u <= args.max_used_mib]
        if not idle and not args.allow_busy_gpu:
            best = max(rows, key=lambda r: r[2] - r[1])
            print(
                f"\nREFUSING TO START: no GPU is idle enough.\n"
                f"  need: >= {args.min_free_vram_gb:.0f}GB free AND <= {args.max_used_mib}MiB "
                f"already in use\n"
                f"  best: GPU{best[0]} with {(best[2]-best[1])/1024:.0f}GB free but "
                f"{best[1]}MiB in use by another process\n"
                f"FLUX.2-dev needs most of a card; sharing one is what made this backend\n"
                f"fail before (OOM mid-run, then a cascade of device-mismatch errors).\n"
                f"Wait for a card to free up, then rerun. Use --allow-busy-gpu to override,\n"
                f"or --max-used-mib to relax the idleness requirement.",
                file=sys.stderr)
            return 3
        if idle:
            chosen = max(idle, key=lambda r: r[2] - r[1])
            os.environ.setdefault("FLUX2_CUDA_DEVICE", str(chosen[0]))
            print(f"Using GPU {chosen[0]} ({(chosen[2]-chosen[1])/1024:.0f}GB free)")
    else:
        print("WARNING: nvidia-smi unavailable — skipping the idle-GPU gate.")

    # Import after FLUX2_CUDA_DEVICE is set: bridge reads FLUX2_* at import time.
    from src import metrics
    from opensource_backends.flux2 import bridge, grid_compose

    if args.variants is not None:
        bridge.FLUX2_NUM_VARIANTS = args.variants
        bridge.FLUX2_BATCH_VARIANTS = min(bridge.FLUX2_BATCH_VARIANTS, args.variants)

    iter_dir = args.reference / f"iter{args.iter}"
    best_script = json.loads((iter_dir / "bestScript.json").read_text(encoding="utf-8"))
    character_description = json.loads(
        (args.reference / "character" / "character_description.json").read_text(encoding="utf-8")
    )
    if args.panels:
        kept = sorted(best_script["clips"], key=lambda c: c["id"])[:args.panels]
        best_script = {**best_script, "clips": kept}

    out_dir = args.out_dir or (REPO_ROOT / "results" / "_flux2_smoke"
                               / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_ids = sorted(c["id"] for c in best_script["clips"])
    print(f"\nPanels: {panel_ids} | variants: {bridge.FLUX2_NUM_VARIANTS} "
          f"| mode: {bridge.FLUX2_OFFLOAD_MODE} | batch: {bridge.FLUX2_BATCH_VARIANTS}")
    print(f"Output: {out_dir}\n")

    try:
        with metrics.stage("flux2_generate_candidates"):
            bridge.generate_candidates(
                best_script, character_description,
                args.target_dir, args.target_dir / "manga_panels", out_dir,
            )

        # Stand in for evaluate_panel_candidates: fixed variant instead of a review.
        with metrics.stage("select_and_compose"):
            from PIL import Image
            selected_dir = out_dir / "selected"
            selected_dir.mkdir(parents=True, exist_ok=True)
            panels = []
            for pid in panel_ids:
                src = out_dir / "candidates" / f"panel{pid}_v{args.pick}.png"
                dst = selected_dir / f"panel{pid}.png"
                dst.write_bytes(src.read_bytes())
                panels.append(Image.open(dst).convert("RGB"))

            sheet = grid_compose.compose_grid(
                panels, columns=3,
                cell_size=(bridge.FLUX2_PANEL_WIDTH, bridge.FLUX2_PANEL_HEIGHT),
            )
            sheet_path = out_dir / "storyboard_0.png"
            sheet.save(sheet_path)
    finally:
        metrics.add("smoke_test", {
            "reference": str(args.reference), "iter": args.iter,
            "panels": panel_ids, "variants": bridge.FLUX2_NUM_VARIANTS,
            "picked_variant": args.pick, "gemini_reviewer": "skipped",
        })
        mpath = metrics.write_metrics(out_dir, target="flux2_smoke",
                                      run_id=out_dir.name)

    m = metrics.collector()
    print(f"\nStoryboard: {sheet_path}  ({sheet.width}x{sheet.height})")
    print(f"Metrics:    {mpath}")
    print(f"Wall clock: {m.payload()['wall_clock_s']}s | Gemini API calls: {m.api_calls} "
          f"(expected 0 — reviewer skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
