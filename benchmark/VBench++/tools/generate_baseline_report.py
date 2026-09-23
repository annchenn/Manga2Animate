#!/usr/bin/env python3
"""Generate a human-readable full report from the baseline segment CSV."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


CASES = ("dualjustice1", "dualjustice2", "elf2", "elf3", "goblin", "kick", "lovehina", "mado")
METHODS = ("storymem", "vimax", "manga2animate")
METRICS = (
    ("i2v_subject", "I2V subj."),
    ("i2v_background", "I2V bg."),
    ("subject_consistency", "Subj. cons."),
    ("background_consistency", "Bg. cons."),
    ("motion_smoothness", "Motion smooth."),
    ("dynamic_degree", "Official DD"),
    ("aesthetic_quality", "Aesthetic"),
    ("imaging_quality", "Imaging"),
    ("modified_dd_mean", "Modified DD"),
    ("spatial_std_mean", "Spatial std"),
    ("temporal_mean_flow_std", "Temporal std"),
    ("normalized_modified_dd_mean", "Norm. modified DD"),
    ("normalized_spatial_std_mean", "Norm. spatial std"),
    ("i2v_subject_dinov3_official", "DINOv3 subj."),
)


def number(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def mean(rows: list[dict], field: str) -> tuple[float | None, int]:
    values = [number(row.get(field, "")) for row in rows]
    valid = [value for value in values if value is not None]
    return (statistics.fmean(valid), len(valid)) if valid else (None, 0)


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    with (run_dir / "summary/baseline_segment_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    lines = [
        "# StoryMem / ViMax / Manga2Animate — Full VBench Report", "",
        "## Completion and interpretation", "",
        f"- Completed rows: {sum(row['status'] == 'success' for row in rows)}/{len(rows)}.",
        "- StoryMem and ViMax each contain 8 cases x 6 segments; Manga2Animate contains dualjustice2 x 6 segments.",
        "- Missing character references are excluded only from original and DINOv3 I2V Subject means.",
        "- Official Dynamic Degree is binary per segment, so its mean is the fraction classified as dynamic.",
        "- Modified DD and its standard deviations are continuous RAFT pixel-flow statistics; normalized columns account for spatial resolution.",
        "- All other official VBench++ columns and DINOv3 Subject are reported on their native approximately 0–1 scales.", "",
        "## Overall method means", "",
    ]
    overall_rows = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        overall_rows.append([
            method, str(len(selected)),
            *[f"{fmt(mean(selected, field)[0])} (n={mean(selected, field)[1]})" for field, _ in METRICS],
        ])
    lines.extend(table(["Method", "Segments", *[label for _, label in METRICS]], overall_rows))

    lines.extend(["", "## Per-case means", ""])
    for case in CASES:
        case_rows = []
        for method in METHODS:
            selected = [row for row in rows if row["case"] == case and row["method"] == method]
            if selected:
                case_rows.append([
                    method,
                    *[f"{fmt(mean(selected, field)[0])} (n={mean(selected, field)[1]})" for field, _ in METRICS],
                ])
        lines.extend([f"### {case}", ""])
        lines.extend(table(["Method", *[label for _, label in METRICS]], case_rows))
        lines.append("")

    lines.extend(["## Overall StoryMem minus ViMax deltas", ""])
    storymem = [row for row in rows if row["method"] == "storymem"]
    vimax = [row for row in rows if row["method"] == "vimax"]
    delta_rows = []
    for field, label in METRICS:
        left, left_n = mean(storymem, field)
        right, right_n = mean(vimax, field)
        delta_rows.append([label, fmt(left), str(left_n), fmt(right), str(right_n), fmt(left - right if left is not None and right is not None else None)])
    lines.extend(table(["Metric", "StoryMem", "n", "ViMax", "n", "StoryMem − ViMax"], delta_rows))

    lines.extend(["", "## Every per-segment value", ""])
    for case in CASES:
        selected = [row for row in rows if row["case"] == case]
        segment_rows = []
        for row in selected:
            segment_rows.append([
                row["method"], row["segment"],
                *[fmt(number(row.get(field, ""))) for field, _ in METRICS],
                row["i2v_subject_status"], row["dinov3_status"], row["status"],
            ])
        lines.extend([f"### {case} — segments", ""])
        lines.extend(table(
            ["Method", "Seg.", *[label for _, label in METRICS], "I2V status", "DINO status", "Overall status"],
            segment_rows,
        ))
        lines.append("")

    lines.extend([
        "## Machine-readable outputs", "",
        "- `summary/baseline_segment_results.csv`: all 102 per-segment rows and every metric.",
        "- `summary/baseline_case_summary.csv`: method-by-case averages and valid counts.",
        "- `summary/baseline_method_summary.csv`: overall method averages and valid counts.",
        "- `summary/baseline_paired_comparison.csv`: 48 paired StoryMem/ViMax segments and per-metric deltas.",
    ])
    (run_dir / "BASELINE_FULL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run_dir / "BASELINE_FULL_REPORT.md")


if __name__ == "__main__":
    main()
