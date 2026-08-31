#!/usr/bin/env python3
"""Combine Manga2Animate, AniMaker, StoryMem, and ViMax evaluation runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("manga2animate", "animaker", "storymem", "vimax")
CASES = ("dualjustice1", "dualjustice2", "elf2", "elf3", "goblin", "kick", "lovehina", "mado")
COMMON_CASES = ("dualjustice1", "elf2", "elf3", "goblin", "kick", "lovehina", "mado")
METRICS = (
    "i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std", "mean_top5_flow",
    "normalized_modified_dd_mean", "normalized_spatial_std_mean", "i2v_subject_dinov3_official",
)
LABELS = {
    "i2v_subject": "I2V Subject", "i2v_background": "I2V Background",
    "subject_consistency": "Subject Cons.", "background_consistency": "Background Cons.",
    "motion_smoothness": "Motion Smooth.", "dynamic_degree": "Official DD",
    "aesthetic_quality": "Aesthetic", "imaging_quality": "Imaging",
    "modified_dd_mean": "Modified DD", "spatial_std_mean": "Spatial std",
    "temporal_mean_flow_std": "Temporal std", "mean_top5_flow": "Mean top-5% flow",
    "normalized_modified_dd_mean": "Normalized DD",
    "normalized_spatial_std_mean": "Normalized spatial std",
    "i2v_subject_dinov3_official": "DINOv3 Subject",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def mean(rows: list[dict], field: str) -> tuple[float | str, int]:
    values = [number(row.get(field)) for row in rows]
    valid = [value for value in values if value is not None]
    return (statistics.fmean(valid), len(valid)) if valid else ("", 0)


def summary(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    keys = sorted(
        {tuple(row[field] for field in group_fields) for row in rows},
        key=lambda key: tuple(
            METHODS.index(value) if value in METHODS else (CASES.index(value) if value in CASES else value)
            for value in key
        ),
    )
    output = []
    for key in keys:
        selected = [row for row in rows if all(row[field] == value for field, value in zip(group_fields, key))]
        item = dict(zip(group_fields, key))
        item["segment_count"] = len(selected)
        for metric in METRICS:
            value, count = mean(selected, metric)
            item[f"{metric}_mean"] = value
            item[f"{metric}_valid_segments"] = count
        output.append(item)
    return output


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def fmt(value) -> str:
    parsed = number(value)
    return "—" if parsed is None else f"{parsed:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    six_path = ROOT / "evaluation_runs/vbenchpp_6case_20260829_015000/summary/segment_results_wide_with_lovehina.csv"
    six_dino_path = ROOT / "evaluation_runs/i2v_subject_dinov3_official_20260829_043000/summary/i2v_subject_dinov3_official_segment_results.csv"
    kick_path = ROOT / "evaluation_runs/kick_20260830_034336/summary/kick_segment_results.csv"
    baseline_path = ROOT / "evaluation_runs/baselines_20260830_172727/summary/baseline_segment_results.csv"
    kick_run = ROOT / "evaluation_runs/kick_20260830_034336"

    dino_rows = read_csv(six_dino_path)
    dino_map = {(row["case"], int(row["segment"]), row["method"]): row for row in dino_rows}
    kick_modified = {}
    for method in ("manga2animate", "animaker"):
        for row in read_csv(kick_run / f"raw/modified_dd_{method}.csv"):
            kick_modified[(row["case"], int(row["segment"]), row["method"])] = row

    combined = []
    for source_path, source_name in (
        (six_path, "vbenchpp_6case_20260829_015000"),
        (kick_path, "kick_20260830_034336"),
        (baseline_path, "baselines_20260830_172727"),
    ):
        for source in read_csv(source_path):
            row = dict(source)
            row["segment"] = str(int(row["segment"]))
            key = (row["case"], int(row["segment"]), row["method"])
            if source_name.startswith("vbenchpp_6case"):
                dino = dino_map[key]
                row["i2v_subject_dinov3_official"] = dino["i2v_subject_dinov3_official"]
                row["dinov3_status"] = dino["status"]
            elif source_name.startswith("kick"):
                modified = kick_modified[key]
                row["normalized_spatial_std_mean"] = modified["normalized_spatial_std_mean"]
            row["source_run"] = source_name
            if not row.get("character_ref") or not Path(row["character_ref"]).is_file():
                row["i2v_subject"] = ""
                row["i2v_subject_dinov3_official"] = ""
                current_status = row.get("i2v_subject_status", "")
                dino_status = row.get("dinov3_status", "")
                row["i2v_subject_status"] = (
                    current_status if current_status.startswith("SKIP_")
                    else dino_status if dino_status.startswith("SKIP_")
                    else "SKIP_NO_CHARACTER"
                )
            row.setdefault("dinov3_status", "SUCCESS" if number(row.get("i2v_subject_dinov3_official")) is not None else row.get("i2v_subject_status", "MISSING"))
            row.setdefault("error", "")
            for field in METRICS:
                row.setdefault(field, "")
            combined.append(row)

    combined.sort(key=lambda row: (CASES.index(row["case"]), int(row["segment"]), METHODS.index(row["method"])))
    fields = [
        "case", "segment", "method", "video_path", "character_ref", "scene_ref",
        *METRICS, "i2v_subject_status", "dinov3_status", "status", "error", "source_run",
    ]
    write_csv(output_dir / "all_methods_segment_results.csv", combined, fields)

    available_summary = summary(combined, ("method",))
    write_csv(output_dir / "all_available_method_summary.csv", available_summary)
    case_summary = summary(combined, ("case", "method"))
    write_csv(output_dir / "all_methods_case_summary.csv", case_summary)

    common = [row for row in combined if row["case"] in COMMON_CASES]
    common_summary = summary(common, ("method",))
    write_csv(output_dir / "common_7case_method_summary.csv", common_summary)

    indexed = {(row["case"], int(row["segment"]), row["method"]): row for row in common}
    wide = []
    for case in COMMON_CASES:
        for segment in range(1, 7):
            item = {"case": case, "segment": segment}
            for method in METHODS:
                row = indexed[(case, segment, method)]
                for metric in METRICS:
                    item[f"{method}_{metric}"] = row[metric]
            wide.append(item)
    write_csv(output_dir / "common_7case_four_method_per_segment.csv", wide)

    failures = [row for row in combined if row.get("status") != "success"]
    missing = {
        method: {metric: len([row for row in combined if row["method"] == method and number(row[metric]) is None]) for metric in METRICS}
        for method in METHODS
    }
    audit = {
        "row_count": len(combined), "success_count": len(combined) - len(failures),
        "failure_count": len(failures),
        "method_row_counts": {method: sum(row["method"] == method for row in combined) for method in METHODS},
        "common_cases": list(COMMON_CASES), "common_rows_per_method": 42,
        "missing_value_counts_by_method": missing,
        "animaker_dualjustice2_status": "not available: no source video/result found",
        "source_files": [str(path.resolve()) for path in (six_path, six_dino_path, kick_path, baseline_path)],
    }
    (output_dir / "combination_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    lines = [
        "# Four-method combined VBench report", "",
        f"- Combined successful per-segment rows: {len(combined) - len(failures)}/{len(combined)}.",
        "- Available coverage: Manga2Animate 8 cases/48 segments; AniMaker 7 cases/42 segments; StoryMem 8 cases/48 segments; ViMax 8 cases/48 segments.",
        "- AniMaker dualjustice2 is not available because no corresponding source videos or results were found.",
        "- Fair four-method scope: 7 shared cases x 6 segments = 42 rows per method; I2V Subject has 35 valid rows per method after identical missing/invalid character-reference exclusions.",
        "- Official Dynamic Degree is a binary-per-segment fraction. Modified DD is continuous RAFT pixel flow; normalized values account for resolution.", "",
        "## Fair comparison: common 7 cases", "",
    ]
    report_metrics = (
        "i2v_subject", "i2v_background", "subject_consistency", "background_consistency",
        "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
        "modified_dd_mean", "normalized_modified_dd_mean", "i2v_subject_dinov3_official",
    )
    fair_rows = []
    for row in common_summary:
        fair_rows.append([
            row["method"], str(row["segment_count"]),
            *[f"{fmt(row[f'{metric}_mean'])} (n={row[f'{metric}_valid_segments']})" for metric in report_metrics],
        ])
    lines.extend(table(["Method", "Segments", *[LABELS[field] for field in report_metrics]], fair_rows))
    lines.extend(["", "## All available data (coverage differs)", ""])
    all_rows = []
    for row in available_summary:
        all_rows.append([
            row["method"], str(row["segment_count"]),
            *[f"{fmt(row[f'{metric}_mean'])} (n={row[f'{metric}_valid_segments']})" for metric in report_metrics],
        ])
    lines.extend(table(["Method", "Segments", *[LABELS[field] for field in report_metrics]], all_rows))
    lines.extend(["", "## Per-case means", ""])
    for case in CASES:
        selected = [row for row in case_summary if row["case"] == case]
        lines.extend([f"### {case}", ""])
        lines.extend(table(
            ["Method", "Segments", *[LABELS[field] for field in report_metrics]],
            [[row["method"], str(row["segment_count"]), *[f"{fmt(row[f'{metric}_mean'])} (n={row[f'{metric}_valid_segments']})" for metric in report_metrics]] for row in selected],
        ))
        lines.append("")
    lines.extend([
        "## Output files", "",
        "- `all_methods_segment_results.csv`: all 186 long-form per-segment rows.",
        "- `all_methods_case_summary.csv`: every available case-method mean and valid count.",
        "- `common_7case_method_summary.csv`: fair four-method aggregate.",
        "- `all_available_method_summary.csv`: aggregate over every available case (coverage differs).",
        "- `common_7case_four_method_per_segment.csv`: one row per shared case/segment, all four methods side by side.",
    ])
    (output_dir / "FOUR_METHOD_COMBINED_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
