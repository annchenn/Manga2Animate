#!/usr/bin/env python3
"""Summarize combined evaluation data under three missing-value policies."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


METHODS = ("manga2animate", "animaker", "storymem", "vimax")
COMMON_CASES = {"dualjustice1", "elf2", "elf3", "goblin", "kick", "lovehina", "mado"}
METRICS = (
    "i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std", "mean_top5_flow",
    "normalized_modified_dd_mean", "normalized_spatial_std_mean", "i2v_subject_dinov3_official",
)


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value) -> str:
    return "—" if value in (None, "") else f"{float(value):.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("combined_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.combined_dir.resolve()
    with (output_dir / "all_methods_segment_results.csv").open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))

    results = []
    scopes = {
        "all_available": all_rows,
        "common_7case": [row for row in all_rows if row["case"] in COMMON_CASES],
    }
    for scope, scope_rows in scopes.items():
        for method in (*METHODS, "all_methods_pooled"):
            selected = scope_rows if method == "all_methods_pooled" else [row for row in scope_rows if row["method"] == method]
            complete = [
                row for row in selected
                if number(row["i2v_subject"]) is not None
                and number(row["i2v_subject_dinov3_official"]) is not None
            ]
            for metric in METRICS:
                values = [number(row[metric]) for row in selected]
                valid = [value for value in values if value is not None]
                complete_values = [number(row[metric]) for row in complete]
                complete_valid = [value for value in complete_values if value is not None]
                results.append({
                    "scope": scope, "method": method, "metric": metric,
                    "all_row_count": len(selected), "valid_value_count": len(valid),
                    "missing_value_count": len(selected) - len(valid),
                    "available_value_mean": statistics.fmean(valid) if valid else "",
                    "missing_as_zero_mean": sum(valid) / len(selected) if selected else "",
                    "complete_reference_row_count": len(complete),
                    "exclude_incomplete_segment_mean": statistics.fmean(complete_valid) if complete_valid else "",
                })
    write_csv(output_dir / "missing_policy_averages_long.csv", results)

    lines = [
        "# Missing-value policy comparison", "",
        "- Available-value mean: missing cells are omitted only for that metric.",
        "- Missing-as-zero mean: missing cells count as zero and remain in the denominator.",
        "- Exclude-incomplete-segment mean: any segment missing original or DINOv3 I2V Subject is removed from every metric.",
        "- `all_available` uses every available case per method; `common_7case` restricts all methods to the same seven cases.", "",
    ]
    for scope in scopes:
        lines.extend([f"## {scope}", ""])
        chosen = [row for row in results if row["scope"] == scope]
        for method in (*METHODS, "all_methods_pooled"):
            rows = [row for row in chosen if row["method"] == method]
            lines.extend([
                f"### {method}", "",
                "| Metric | Rows | Valid | Missing | Available mean | Missing=0 mean | Complete rows | Exclude incomplete segment mean |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for row in rows:
                lines.append(
                    f"| {row['metric']} | {row['all_row_count']} | {row['valid_value_count']} | {row['missing_value_count']} | "
                    f"{fmt(row['available_value_mean'])} | {fmt(row['missing_as_zero_mean'])} | "
                    f"{row['complete_reference_row_count']} | {fmt(row['exclude_incomplete_segment_mean'])} |"
                )
            lines.append("")
    (output_dir / "MISSING_POLICY_AVERAGES.md").write_text("\n".join(lines) + "\n")
    print(output_dir / "MISSING_POLICY_AVERAGES.md")


if __name__ == "__main__":
    main()
