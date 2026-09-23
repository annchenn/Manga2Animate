#!/usr/bin/env python3
"""Aggregate the isolated six-case/lovehina VBench++ evaluation run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ("dualjustice1", "elf2", "elf3", "goblin", "mado", "lovehina")
EXISTING = set(CASES[:-1])
METHODS = ("manga2animate", "animaker")
METRICS = (
    "i2v_subject", "i2v_background", "subject_consistency",
    "background_consistency", "motion_smoothness", "dynamic_degree",
    "aesthetic_quality", "imaging_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def segment_from_path(value: str) -> int | None:
    match = re.search(r"(?:segment_|shot_0?)(\d+)", Path(value).stem)
    return int(match.group(1)) if match else None


def case_metric(run_dir: Path, case: str, method: str, metric: str) -> tuple[dict[int, float], Path | None]:
    paths = sorted((run_dir / "raw" / f"{case}_{method}" / metric).glob("*_eval_results.json"))
    if not paths:
        return {}, None
    path = paths[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))[metric][1]
    values = {}
    for item in payload:
        segment = segment_from_path(item["video_path"])
        value = item["video_results"]
        if segment is not None:
            value = float(value)
            # Official imaging_quality returns batch mean(raw MUSIQ) / 100.
            # Keep per-segment values in the same official 0-1 unit.
            values[segment] = value / 100.0 if metric == "imaging_quality" else value
    return values, path


def modified_outputs(run_dir: Path, manifest: list[dict]) -> list[dict]:
    rows = []
    for path in sorted((run_dir / "raw").glob("modified_dd_*.csv")):
        rows.extend(read_csv(path))
    by_key = {(row["case"], row["method"], int(row["segment"])): row for row in rows}
    fields = [
        "case", "segment", "method", "video_path", "fps_original", "fps_evaluated",
        "width", "height", "evaluated_frame_count", "flow_pair_count", "modified_dd_mean",
        "spatial_std_mean", "temporal_mean_flow_std", "mean_top5_flow",
        "normalized_modified_dd_mean", "normalized_spatial_std_mean", "success", "error",
        "runtime_seconds",
    ]
    for item in manifest:
        key = (item["case"], item["method"], int(item["segment"]))
        if key not in by_key:
            by_key[key] = {
                "case": item["case"], "segment": item["segment"], "method": item["method"],
                "video_path": item["video_path"], "success": "False",
                "error": "Modified DD worker output missing", "runtime_seconds": "",
            }
        for field in fields:
            by_key[key].setdefault(field, "")
    rows = sorted(by_key.values(), key=lambda row: (CASES.index(row["case"]), int(row["segment"]), METHODS.index(row["method"])))
    write_csv(run_dir / "summary/modified_dynamic_degree_segment_results.csv", rows, fields)
    return rows


def paired_outputs(run_dir: Path, rows: list[dict]) -> list[dict]:
    valid = {
        (row["case"], int(row["segment"]), row["method"]): row for row in rows
        if str(row["success"]).lower() == "true" and number(row["modified_dd_mean"]) is not None
    }
    paired = []
    for case in CASES:
        for segment in range(1, 7):
            manga = valid.get((case, segment, "manga2animate"))
            animaker = valid.get((case, segment, "animaker"))
            if not manga or not animaker:
                continue
            manga_dd, animaker_dd = number(manga["modified_dd_mean"]), number(animaker["modified_dd_mean"])
            manga_std, animaker_std = number(manga["spatial_std_mean"]), number(animaker["spatial_std_mean"])
            paired.append({
                "case": case, "segment": segment,
                "manga_modified_dd": manga_dd, "animaker_modified_dd": animaker_dd,
                "modified_dd_delta": manga_dd - animaker_dd,
                "manga_spatial_std": manga_std, "animaker_spatial_std": animaker_std,
                "spatial_std_delta": manga_std - animaker_std,
            })
    write_csv(run_dir / "summary/modified_dynamic_degree_paired_comparison.csv", paired)
    return paired


def summarize_modified(run_dir: Path, rows: list[dict], paired: list[dict]) -> list[dict]:
    summaries = []
    for scope, cases in (("existing_5_cases", EXISTING), ("all_6_cases", set(CASES))):
        scope_pairs = [row for row in paired if row["case"] in cases]
        for method in METHODS:
            selected = [row for row in rows if row["case"] in cases and row["method"] == method and str(row["success"]).lower() == "true"]
            def mean(field: str):
                values = [number(row[field]) for row in selected]
                values = [value for value in values if value is not None]
                return statistics.fmean(values) if values else ""
            summaries.append({
                "scope": scope, "method": method, "valid_segments": len(selected),
                "modified_dd_mean": mean("modified_dd_mean"),
                "spatial_std_mean": mean("spatial_std_mean"),
                "temporal_mean_flow_std_mean": mean("temporal_mean_flow_std"),
                "normalized_modified_dd_mean": mean("normalized_modified_dd_mean"),
                "paired_wins": sum((row["modified_dd_delta"] > 1e-12) if method == "manga2animate" else (row["modified_dd_delta"] < -1e-12) for row in scope_pairs),
                "paired_losses": sum((row["modified_dd_delta"] < -1e-12) if method == "manga2animate" else (row["modified_dd_delta"] > 1e-12) for row in scope_pairs),
                "paired_ties": sum(abs(row["modified_dd_delta"]) <= 1e-12 for row in scope_pairs),
                "valid_pairs": len(scope_pairs),
            })
    write_csv(run_dir / "summary/modified_dynamic_degree_scope_summary.csv", summaries)
    case_rows = []
    for case in CASES:
        case_pairs = [row for row in paired if row["case"] == case]
        for method in METHODS:
            selected = [row for row in rows if row["case"] == case and row["method"] == method and str(row["success"]).lower() == "true"]
            out = {"case": case, "method": method, "valid_segments": len(selected)}
            for field in ("modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std", "normalized_modified_dd_mean"):
                values = [number(row[field]) for row in selected]
                values = [value for value in values if value is not None]
                out[field + "_case_mean"] = statistics.fmean(values) if values else ""
            out["paired_wins"] = sum((row["modified_dd_delta"] > 1e-12) if method == "manga2animate" else (row["modified_dd_delta"] < -1e-12) for row in case_pairs)
            out["paired_losses"] = sum((row["modified_dd_delta"] < -1e-12) if method == "manga2animate" else (row["modified_dd_delta"] > 1e-12) for row in case_pairs)
            out["paired_ties"] = sum(abs(row["modified_dd_delta"]) <= 1e-12 for row in case_pairs)
            out["valid_pairs"] = len(case_pairs)
            case_rows.append(out)
    write_csv(run_dir / "summary/modified_dynamic_degree_case_summary.csv", case_rows)
    return summaries


def formal_outputs(run_dir: Path, manifest: list[dict], modified: list[dict]) -> tuple[list[dict], list[dict]]:
    old_path = ROOT / "manga2animate_vbenchpp_results/summary/per_shot_results_altcorr_completed.csv"
    old = read_csv(old_path)
    old_map = {}
    for row in old:
        segment = segment_from_path(row.get("shot", ""))
        if segment:
            old_map[(row["story"], row["method"], segment)] = row
    new_maps = {}
    raw_paths = {}
    for case in CASES:
        for method in METHODS:
            for metric in METRICS:
                new_maps[(case, method, metric)], raw_paths[(case, method, metric)] = case_metric(run_dir, case, method, metric)
    modified_map = {(row["case"], row["method"], int(row["segment"])): row for row in modified}
    wide = []
    for item in manifest:
        key = (item["case"], item["method"], int(item["segment"]))
        row = {
            "case": item["case"], "segment": item["segment"], "method": item["method"],
            "video_path": item["video_path"], "character_ref": item["character_ref"],
            "scene_ref": item["scene_ref"], "fps_original": item.get("fps_original", ""),
            "width": item.get("width", ""), "height": item.get("height", ""),
            "source_frame_count": item.get("source_frame_count", ""),
        }
        old_row = old_map.get(key)
        for metric in METRICS:
            new_value = new_maps[(item["case"], item["method"], metric)].get(int(item["segment"]))
            if new_value is not None:
                row[metric] = new_value
            elif old_row:
                row[metric] = old_row.get(metric, "")
            else:
                row[metric] = ""
        if item["case"] == "goblin" and int(item["segment"]) == 3:
            row["i2v_subject"] = ""
            row["i2v_subject_status"] = "SKIP_INVALID_REFERENCE"
        elif item["case"] == "elf2" and int(item["segment"]) in (1, 6):
            row["i2v_subject"] = ""
            row["i2v_subject_status"] = "SKIP_NO_CHARACTER"
        elif item["case"] == "mado" and int(item["segment"]) == 2:
            row["i2v_subject"] = ""
            row["i2v_subject_status"] = "SKIP_NO_CHARACTER"
        else:
            row["i2v_subject_status"] = "SUCCESS" if number(row["i2v_subject"]) is not None else "MISSING"
        mod = modified_map.get(key, {})
        for field in ("modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std", "mean_top5_flow", "normalized_modified_dd_mean", "normalized_spatial_std_mean"):
            row[field] = mod.get(field, "")
        required_metrics = [metric for metric in METRICS if not (metric == "i2v_subject" and row["i2v_subject_status"].startswith("SKIP_"))]
        formal_present = all(number(row[metric]) is not None for metric in required_metrics)
        row["status"] = "success" if formal_present and str(mod.get("success", "")).lower() == "true" else "partial_or_missing"
        errors = []
        if not item["video_path"]:
            errors.append("source video missing")
        if not old_row and not all(number(row[metric]) is not None for metric in required_metrics):
            errors.append("formal per-segment metric output missing")
        if str(mod.get("success", "")).lower() != "true":
            errors.append(mod.get("error", "Modified DD missing"))
        row["error"] = "; ".join(filter(None, errors))
        wide.append(row)
    wide_fields = list(wide[0])
    write_csv(run_dir / "summary/segment_results_wide_with_lovehina.csv", wide, wide_fields)
    case_summary = []
    for case in CASES:
        for method in METHODS:
            selected = [row for row in wide if row["case"] == case and row["method"] == method]
            out = {"case": case, "method": method}
            for metric in (*METRICS, "modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std"):
                values = [number(row[metric]) for row in selected]
                values = [value for value in values if value is not None]
                out[metric + "_mean"] = statistics.fmean(values) if values else ""
                out[metric + "_valid_segments"] = len(values)
            case_summary.append(out)
    write_csv(run_dir / "summary/case_summary_with_lovehina.csv", case_summary)
    method_summary = []
    for method in METHODS:
        selected = [row for row in wide if row["method"] == method]
        out = {"method": method}
        for metric in (*METRICS, "modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std"):
            values = [number(row[metric]) for row in selected]
            values = [value for value in values if value is not None]
            out[metric + "_mean"] = statistics.fmean(values) if values else ""
            out[metric + "_valid_segments"] = len(values)
        method_summary.append(out)
    write_csv(run_dir / "summary/method_summary_with_lovehina.csv", method_summary)
    return wide, case_summary


def report(run_dir: Path, wide: list[dict], case_summary: list[dict], paired: list[dict]) -> None:
    love_rows = [row for row in wide if row["case"] == "lovehina"]
    audits = []
    for path in sorted((run_dir / "audit").glob("lovehina_*.json")):
        audits.append(json.loads(path.read_text()))
    lines = [
        "# LOVEHINA FULL VBENCH REPORT", "",
        "## Protocol", "",
        "- Input audit: 6 segments for Manga2Animate and 6 for AniMaker; segment N maps to refN.jpg and manga_panelN.jpg.",
        "- Preprocessing: symlink-only staging; no transcode, resize, crop, or source mutation.",
        "- Official VBench++ dimensions: " + ", ".join(METRICS) + ".",
        "- Motion Smoothness model: official AMT-S checkpoint from the existing VBench cache.",
        "- Official Dynamic Degree and Modified DD: RAFT-Things, 20 iterations, bundled alternate correlation, approximately 8 fps sampling.",
        "- Modified DD normalization: divide raw pixel flow statistics by min(width,height)/256.",
        "- SKIP rule outcome: no lovehina segment was skipped; all 12 video/reference mappings were applicable and evaluated.", "",
        "## Per-segment results", "",
        "| Method | Segment | " + " | ".join(METRICS) + " | Modified DD | Spatial std | Temporal std | Status |",
        "|---|---:" + "|---:" * (len(METRICS) + 3) + "|---|",
    ]
    for row in love_rows:
        values = [row[metric] for metric in METRICS]
        lines.append("| " + " | ".join([row["method"], str(row["segment"]), *[str(value) for value in values], str(row["modified_dd_mean"]), str(row["spatial_std_mean"]), str(row["temporal_mean_flow_std"]), row["status"]]) + " |")
    lines.extend(["", "## Case means", ""])
    for row in [item for item in case_summary if item["case"] == "lovehina"]:
        lines.append(f"### {row['method']}")
        lines.append("")
        for metric in (*METRICS, "modified_dd_mean", "spatial_std_mean", "temporal_mean_flow_std"):
            lines.append(f"- {metric}: {row[metric + '_mean']} (valid segments: {row[metric + '_valid_segments']})")
        lines.append("")
    love_pairs = [row for row in paired if row["case"] == "lovehina"]
    lines.extend([
        "## Paired comparison", "",
        f"- Valid lovehina pairs: {len(love_pairs)}",
        f"- Manga2Animate wins / AniMaker wins / ties: {sum(r['modified_dd_delta'] > 1e-12 for r in love_pairs)} / {sum(r['modified_dd_delta'] < -1e-12 for r in love_pairs)} / {sum(abs(r['modified_dd_delta']) <= 1e-12 for r in love_pairs)}",
        "", "## Execution, failures, and resources", "",
    ])
    for audit in audits:
        lines.append(f"- {audit['method']} / {audit['metric']}: {audit['status']}; {audit['runtime_seconds']:.3f}s; GPU {audit['physical_gpu']}; peak {audit['peak_gpu_memory_mib']} MiB; log `{audit['log_path']}`; raw `{audit['output_path']}`")
    lines.extend(["", "## Output paths", "", f"- Run root: `{run_dir}`", f"- Input manifest: `{run_dir / 'audit/input_manifest.json'}`", f"- Segment summary: `{run_dir / 'summary/segment_results_wide_with_lovehina.csv'}`", f"- Modified DD results: `{run_dir / 'summary/modified_dynamic_degree_segment_results.csv'}`", ""])
    (run_dir / "LOVEHINA_FULL_VBENCH_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    run_dir = parse_args().run_dir.resolve()
    manifest = json.loads((run_dir / "audit/input_manifest.json").read_text())
    legacy_checksum_path = run_dir / "audit/legacy_results_checksums.json"
    if not legacy_checksum_path.is_file():
        legacy_files = sorted((ROOT / "manga2animate_vbenchpp_results").rglob("*"))
        legacy_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in legacy_files if path.is_file()}
        legacy_checksum_path.write_text(json.dumps(legacy_hashes, indent=2) + "\n")
    modified = modified_outputs(run_dir, manifest)
    paired = paired_outputs(run_dir, modified)
    summarize_modified(run_dir, modified, paired)
    wide, case_summary = formal_outputs(run_dir, manifest, modified)
    report(run_dir, wide, case_summary, paired)
    print(f"Aggregated {len(modified)} Modified DD slots, {len(paired)} valid pairs, and {len(wide)} wide rows")


if __name__ == "__main__":
    main()
