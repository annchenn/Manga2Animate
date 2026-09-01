#!/usr/bin/env python3
"""Aggregate the isolated six-case timm DINOv3 I2V Subject run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


CASES = ("dualjustice1", "elf2", "elf3", "goblin", "mado", "lovehina")
METHODS = ("manga2animate", "animaker")
SCORE = "i2v_subject_dinov3_timm"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict) -> float | None:
    try:
        result = float(row[SCORE])
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    rows = read_csv(run_dir / "raw/manga2animate.csv") + read_csv(run_dir / "raw/animaker.csv")
    rows.sort(key=lambda row: (CASES.index(row["case"]), int(row["segment"]), METHODS.index(row["method"])))
    write_csv(run_dir / "summary/i2v_subject_dinov3_timm_segment_results.csv", rows)

    case_rows = []
    for case in CASES:
        for method in METHODS:
            chosen = [value(row) for row in rows if row["case"] == case and row["method"] == method]
            valid = [item for item in chosen if item is not None]
            case_rows.append({
                "case": case, "method": method,
                "i2v_subject_dinov3_timm_mean": float(np.mean(valid)),
                "i2v_subject_dinov3_timm_std": float(np.std(valid)),
                "valid_segment_count": len(valid), "skip_segment_count": len(chosen) - len(valid),
            })
    write_csv(run_dir / "summary/i2v_subject_dinov3_timm_case_summary.csv", case_rows)

    method_rows = []
    for method in METHODS:
        valid = [value(row) for row in rows if row["method"] == method and value(row) is not None]
        method_rows.append({
            "method": method, "i2v_subject_dinov3_timm_mean": float(np.mean(valid)),
            "i2v_subject_dinov3_timm_std": float(np.std(valid)), "valid_segment_count": len(valid),
        })
    write_csv(run_dir / "summary/i2v_subject_dinov3_timm_method_summary.csv", method_rows)

    indexed = {(row["case"], int(row["segment"]), row["method"]): row for row in rows}
    paired = []
    for case in CASES:
        for segment in range(1, 7):
            manga = value(indexed[(case, segment, "manga2animate")])
            animaker = value(indexed[(case, segment, "animaker")])
            if manga is None or animaker is None:
                continue
            delta = manga - animaker
            paired.append({
                "case": case, "segment": segment,
                "manga_i2v_subject_dinov3_timm": manga,
                "animaker_i2v_subject_dinov3_timm": animaker,
                "delta_manga_minus_animaker": delta,
                "winner": "Manga2Animate" if delta > 1e-12 else ("AniMaker" if delta < -1e-12 else "Tie"),
            })
    write_csv(run_dir / "summary/i2v_subject_dinov3_timm_paired_comparison.csv", paired)

    manga_mean = float(method_rows[0]["i2v_subject_dinov3_timm_mean"])
    animaker_mean = float(method_rows[1]["i2v_subject_dinov3_timm_mean"])
    wins = {name: sum(row["winner"] == name for row in paired) for name in ("Manga2Animate", "AniMaker", "Tie")}
    report = [
        "# I2V Subject — DINOv3 timm Trial", "",
        "- Backbone: `timm/vit_base_patch16_dinov3_qkvb.lvd1689m` (ViT-B/16, LVD-1689M).",
        "- Implementation: timm 1.0.20; 768-dimensional normalized image embeddings.",
        "- Preprocessing: 256x256 bicubic center transform, ImageNet mean/std, as resolved by timm model config.",
        "- Score formula retained from VBench++: 0.4 max reference similarity + 0.3 mean consecutive similarity + 0.3 minimum consecutive similarity.",
        "- Every source video frame was evaluated; no temporal subsampling.",
        "- Existing SKIP rules retained: elf2 segments 1 and 6 (no character), mado segment 2 (no character), goblin segment 3 (invalid reference/result).", "",
        "## Overall", "",
        f"- Manga2Animate: {manga_mean:.12f} (32 valid segments).",
        f"- AniMaker: {animaker_mean:.12f} (32 valid segments).",
        f"- Delta (Manga2Animate - AniMaker): {manga_mean - animaker_mean:.12f}.",
        f"- Paired wins/losses/ties for Manga2Animate: {wins['Manga2Animate']}/{wins['AniMaker']}/{wins['Tie']} across {len(paired)} valid pairs.", "",
        "## Case means", "",
        "| Case | Manga2Animate | AniMaker | Delta |", "|---|---:|---:|---:|",
    ]
    for case in CASES:
        manga = next(float(row["i2v_subject_dinov3_timm_mean"]) for row in case_rows if row["case"] == case and row["method"] == "manga2animate")
        ani = next(float(row["i2v_subject_dinov3_timm_mean"]) for row in case_rows if row["case"] == case and row["method"] == "animaker")
        report.append(f"| {case} | {manga:.6f} | {ani:.6f} | {manga - ani:+.6f} |")
    report.extend(["", "## Outputs", "",
        "- `summary/i2v_subject_dinov3_timm_segment_results.csv`",
        "- `summary/i2v_subject_dinov3_timm_case_summary.csv`",
        "- `summary/i2v_subject_dinov3_timm_method_summary.csv`",
        "- `summary/i2v_subject_dinov3_timm_paired_comparison.csv`", "",
        "This is a custom DINOv3 replacement trial, not the original official VBench++ DINO-v1 I2V Subject metric.",
    ])
    (run_dir / "I2V_SUBJECT_DINOV3_TIMM_REPORT.md").write_text("\n".join(report) + "\n")
    audit = {
        "row_count": len(rows), "success_count": sum(row["status"] == "SUCCESS" for row in rows),
        "skip_count": sum(row["status"].startswith("SKIP_") for row in rows),
        "paired_count": len(paired), "manga_mean": manga_mean, "animaker_mean": animaker_mean,
        "paired_wins": wins,
    }
    (run_dir / "audit/aggregation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
