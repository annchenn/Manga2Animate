#!/usr/bin/env python3
"""Render a bench_results run into a readable result.md.

Called automatically by run_bench.sh; can also be run by hand:

    python src/make_result_md.py                          # newest run
    python src/make_result_md.py --run data/bench_results/ours/storyboard/en/20260807_112929
    python src/make_result_md.py --output /tmp/r.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = BENCH_DIR / "data" / "bench_results"

ALL_METRICS = ["cids", "csd", "aesthetic", "diversity", "prompt_align"]

# metric key -> (display label, higher is better)
METRIC_LABELS = {
    "cids_self_mean": "CIDS self",
    "cids_cross_mean": "CIDS cross",
    "copy_paste_score": "Copy-paste",
    "occm": "OCCM",
    "single_character_action": "Single character action (VLM)",
    "self_csd": "CSD self",
    "cross_csd": "CSD cross",
    "aesthetic_score": "Aesthetic",
    "inception_score": "Inception Score",
    "inception_score_std": "IS std",
    "scene": "PromptAlign scene (VLM)",
    "character_action": "PromptAlign character action (VLM)",
    "camera": "PromptAlign camera (VLM)",
}


def find_latest_run(results_root: Path) -> Path | None:
    """Newest directory containing a metadata.json, by name (timestamps sort lexically)."""
    runs = [p.parent for p in results_root.glob("*/*/*/*/metadata.json")]
    return max(runs, key=lambda p: p.name) if runs else None


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, int):
        return str(v)
    return "—" if v is None else str(v)


def render(run_dir: Path) -> str:
    meta = load_json(run_dir / "metadata.json") or {}
    run = meta.get("run", {})
    summary = load_json(run_dir / "summary.json") or {}
    manifest = load_json(run_dir / "manifest.json") or {}
    metrics_dir = run_dir / "metrics"

    L: list[str] = []
    L.append("# ViStoryBench Results")
    L.append("")
    L.append(f"**Run:** `{run_dir}`")
    L.append("")

    # --- Run info ---
    L.append("| field | value |")
    L.append("|---|---|")
    for k in ("method", "mode", "language", "dataset_name", "timestamp", "outputs_timestamp"):
        if k in run:
            L.append(f"| {k} | `{run[k]}` |")
    L.append("")

    # --- Which stories were evaluated ---
    stories = set()
    for m in manifest.get("metrics", {}).values():
        stories.update(m.keys())
    for m in ALL_METRICS:
        d = load_json(metrics_dir / m / "story_results.json")
        if isinstance(d, dict):
            stories.update(d.keys())
    if stories:
        L.append(f"**Stories evaluated:** {', '.join(f'`{s}`' for s in sorted(stories))}")
        L.append("")

    # --- Headline table, laid out like the ViStoryBench paper ---
    smetrics = summary.get("metrics", {})
    ran = [m for m in ALL_METRICS if smetrics.get(m)]

    def val(metric: str, key: str):
        block = smetrics.get(metric) or {}
        return (block.get("metrics") or {}).get(key)

    pa = [val("prompt_align", "scene"), val("prompt_align", "camera"),
          val("prompt_align", "character_action"), val("cids", "single_character_action")]
    pa_nums = [v for v in pa if isinstance(v, (int, float))]
    pa_avg = sum(pa_nums) / len(pa_nums) if pa_nums else None

    cells = [
        run.get("method", "—"), run.get("mode", "—"),
        val("csd", "cross_csd"), val("csd", "self_csd"),
        val("cids", "cids_cross_mean"), val("cids", "cids_self_mean"),
        *pa, pa_avg,
        val("cids", "occm"),
        val("diversity", "inception_score"),
        val("aesthetic", "aesthetic_score"),
        val("cids", "copy_paste_score"),
    ]

    # Grouped vertical layout: the group label is printed once, like the merged
    # header cells in the paper's table.
    groups = [
        ("CSD↑", [("Cross", val("csd", "cross_csd")), ("Self", val("csd", "self_csd"))]),
        ("CIDS↑", [("Cross", val("cids", "cids_cross_mean")),
                   ("Self", val("cids", "cids_self_mean"))]),
        ("PA↑", [("Scene", pa[0]), ("Shot", pa[1]), ("CI", pa[2]), ("IA", pa[3]),
                 ("Avg.", pa_avg)]),
        ("CM↑", [("", val("cids", "occm"))]),
        ("Inc↑", [("", val("diversity", "inception_score"))]),
        ("Aes↑", [("", val("aesthetic", "aesthetic_score"))]),
        ("CP", [("", val("cids", "copy_paste_score"))]),
    ]

    L.append("## Summary")
    L.append("")
    L.append("| Metric | | Value |")
    L.append("|---|---|---|")
    for label, rows in groups:
        for i, (sub, v) in enumerate(rows):
            L.append(f"| {'**' + label + '**' if i == 0 else ''} | {sub} | {fmt(v)} |")
    L.append("")
    L.append("PA: Scene=`scene`, Shot=`camera`, CI=`character_action`, "
             "IA=`single_character_action`. CM=`occm`, Inc=Inception Score, CP=copy-paste.")
    L.append("")

    if not ran:
        L.append("No metrics recorded.")
        L.append("")

    # --- Not run ---
    missing = [m for m in ALL_METRICS if m not in ran]
    if missing:
        L.append("**Not run:** " + ", ".join(f"`{m}`" for m in missing))
        L.append("")

    # --- CIDS per character ---
    cids = load_json(metrics_dir / "cids" / "story_results.json")
    if isinstance(cids, dict):
        rows = []
        for sid, rec in sorted(cids.items()):
            for char, sc in (rec.get("cids") or {}).items():
                if isinstance(sc, dict):
                    rows.append((sid, char, sc.get("self"), sc.get("cross")))
        if rows:
            L.append("## CIDS per character")
            L.append("")
            L.append("| story | character | self | cross |")
            L.append("|---|---|---|---|")
            for sid, char, s, c in rows:
                L.append(f"| `{sid}` | {char} | {fmt(s)} | {fmt(c)} |")
            L.append("")

        # --- CIDS per shot ---
        shot_rows = []
        chars: list[str] = []
        for sid, rec in sorted(cids.items()):
            shot_keys = [k for k in rec if k.startswith("shot-") and k.split("-", 1)[1].isdigit()]
            for k in sorted(shot_keys, key=lambda x: int(x.split("-", 1)[1])):
                per = rec[k]
                if not isinstance(per, dict):
                    continue
                for ch in per:
                    if ch not in chars:
                        chars.append(ch)
                shot_rows.append((sid, k.split("-", 1)[1], per))
        if shot_rows:
            L.append("## CIDS per shot (cross similarity)")
            L.append("")
            L.append("| story | shot | " + " | ".join(chars) + " |")
            L.append("|---|---|" + "---|" * len(chars))
            for sid, idx, per in sorted(shot_rows, key=lambda r: (r[0], int(r[1]))):
                cells = []
                for ch in chars:
                    v = per.get(ch)
                    cells.append(fmt(v.get("cross_sim")) if isinstance(v, dict) else "—")
                L.append(f"| `{sid}` | {idx} | " + " | ".join(cells) + " |")
            L.append("")

    # --- Aesthetic per image ---
    aes = load_json(metrics_dir / "aesthetic" / "story_results.json")
    if isinstance(aes, dict):
        rows = []
        for sid, rec in sorted(aes.items()):
            per = rec.get("per_image_scores")
            entries = per.items() if isinstance(per, dict) else enumerate(per or [], 1)
            for key, val in entries:
                # Entries are either {"shot_index": N, "score": X} or a bare number.
                if isinstance(val, dict):
                    rows.append((sid, val.get("shot_index", key), val.get("score")))
                else:
                    rows.append((sid, key, val))
        if rows:
            L.append("## Aesthetic per shot")
            L.append("")
            L.append("| story | shot | score |")
            L.append("|---|---|---|")
            for sid, k, v in rows:
                L.append(f"| `{sid}` | {Path(str(k)).name} | {fmt(v)} |")
            L.append("")

    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render bench_results into result.md")
    ap.add_argument("--run", default=None, help="Run directory. Default: newest under results root")
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    ap.add_argument("--output", default=None, help="Default: <run>/result.md")
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run)
    else:
        run_dir = find_latest_run(Path(args.results_root))
        if run_dir is None:
            print(f"ERROR: no runs found under {args.results_root}", file=sys.stderr)
            return 1
    if not (run_dir / "metadata.json").exists():
        print(f"ERROR: {run_dir} is not a bench run (no metadata.json)", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else run_dir / "result.md"
    out.write_text(render(run_dir), encoding="utf-8")
    print(f"\033[92mWrote {out}\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
