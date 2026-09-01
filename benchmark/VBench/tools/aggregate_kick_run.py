#!/usr/bin/env python3
"""Aggregate kick VBench++, continuous Dynamic Degree, and DINOv3 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


METHODS = ("manga2animate", "animaker")
METRICS = (
    "i2v_subject", "i2v_background", "subject_consistency",
    "background_consistency", "motion_smoothness", "dynamic_degree",
    "aesthetic_quality", "imaging_quality",
)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def segment(value: str) -> int | None:
    match = re.search(r"(?:segment_|shot_0?)(\d+)", Path(value).stem)
    return int(match.group(1)) if match else None


def formal_values(run_dir: Path, method: str, metric: str) -> dict[int, float]:
    paths = sorted((run_dir / "raw" / f"kick_{method}" / metric).glob("*_eval_results.json"))
    if not paths:
        return {}
    items = json.loads(paths[-1].read_text())[metric][1]
    result = {}
    for item in items:
        index = segment(item["video_path"])
        if index is not None:
            value = float(item["video_results"])
            result[index] = value / 100.0 if metric == "imaging_quality" else value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "audit/input_manifest.json").read_text())
    formal = {(method, metric): formal_values(run_dir, method, metric) for method in METHODS for metric in METRICS}
    modified_rows = read_csv(run_dir / "raw/modified_dd_manga2animate.csv") + read_csv(run_dir / "raw/modified_dd_animaker.csv")
    modified = {(row["method"], int(row["segment"])): row for row in modified_rows}
    dino_rows = read_csv(run_dir / "raw/dinov3_manga2animate.csv") + read_csv(run_dir / "raw/dinov3_animaker.csv")
    dino = {(row["method"], int(row["segment"])): row for row in dino_rows}

    rows = []
    for item in manifest:
        method, index = item["method"], int(item["segment"])
        row = {
            "case": "kick", "segment": index, "method": method,
            "video_path": item["video_path"], "character_ref": item["character_ref"],
            "scene_ref": item["scene_ref"], "i2v_subject_status": item["i2v_subject_status"],
        }
        for metric in METRICS:
            row[metric] = formal[(method, metric)].get(index, "")
        mod = modified.get((method, index), {})
        for field in ("modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std", "mean_top5_flow", "normalized_modified_dd_mean"):
            row[field] = mod.get(field, "")
        dino_row = dino.get((method, index), {})
        row["i2v_subject_dinov3_official"] = dino_row.get("i2v_subject_dinov3_official", "")
        row["dinov3_status"] = dino_row.get("status", item["i2v_subject_status"] if not item["character_ref"] else "MISSING")
        required = [metric for metric in METRICS if metric != "i2v_subject" or item["character_ref"]]
        row["status"] = "success" if (
            all(number(row[name]) is not None for name in required)
            and number(row["modified_dd_mean"]) is not None
            and (not item["character_ref"] or number(row["i2v_subject_dinov3_official"]) is not None)
        ) else "partial_or_missing"
        rows.append(row)
    rows.sort(key=lambda row: (int(row["segment"]), METHODS.index(row["method"])))
    write_csv(run_dir / "summary/kick_segment_results.csv", rows)

    means = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        output = {"case": "kick", "method": method, "segment_count": len(selected)}
        for metric in (*METRICS, "modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std", "i2v_subject_dinov3_official"):
            values = [number(row[metric]) for row in selected]
            values = [value for value in values if value is not None]
            output[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            output[f"{metric}_valid_segments"] = len(values)
        means.append(output)
    write_csv(run_dir / "summary/kick_method_summary.csv", means)
    audit = {
        "segment_rows": len(rows), "successful_rows": sum(row["status"] == "success" for row in rows),
        "partial_or_missing_rows": sum(row["status"] != "success" for row in rows),
        "expected_i2v_subject_skips": sum(not row["character_ref"] for row in rows),
    }
    (run_dir / "audit/aggregation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
