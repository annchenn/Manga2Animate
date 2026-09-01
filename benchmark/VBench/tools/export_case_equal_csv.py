#!/usr/bin/env python3
"""Export full-precision, case-equal method summaries from final per-case data."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


METHOD_ORDER = ("manga2animate", "animaker", "storymem", "vimax")
METHOD_LABELS = {
    "manga2animate": "Manga2Animate (Our Pipeline)",
    "animaker": "AniMaker",
    "storymem": "StoryMem",
    "vimax": "ViMax",
}
COMMON_7_CASES = (
    "dualjustice1",
    "elf2",
    "elf3",
    "goblin",
    "kick",
    "lovehina",
    "mado",
)
CASE_ORDER = (
    "dualjustice1",
    "dualjustice2",
    "elf2",
    "elf3",
    "goblin",
    "kick",
    "lovehina",
    "mado",
)

# Output order follows Manga2Animate_VBench_Final_CaseEqual_Formula.xlsx.
METRICS = (
    ("i2v_subject", "i2v_subject_mean"),
    ("i2v_background", "i2v_background_mean"),
    ("subject_consistency", "subject_consistency_mean"),
    ("background_consistency", "background_consistency_mean"),
    ("motion_smoothness", "motion_smoothness_mean"),
    ("aesthetic_quality", "aesthetic_quality_mean"),
    ("imaging_quality", "imaging_quality_mean"),
    ("dinov3_i2v_subject_score", "i2v_subject_dinov3_official_mean"),
    ("official_dynamic_degree", "dynamic_degree_mean"),
    ("modified_dynamic_degree", "modified_dd_mean_mean"),
    ("spatial_standard_deviation", "spatial_std_mean_mean"),
    ("temporal_standard_deviation", "temporal_mean_flow_std_mean"),
    ("normalized_dynamic_degree", "normalized_modified_dd_mean_mean"),
    ("normalized_spatial_standard_deviation", "normalized_spatial_std_mean_mean"),
)


def parse_number(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def number_text(value: float) -> str:
    # repr() is the shortest decimal that round-trips to the same binary64.
    return repr(value)


def summarize(rows: list[dict[str, str]], scope: str) -> list[dict[str, str | int]]:
    output: list[dict[str, str | int]] = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        if scope == "common_7_cases_including_goblin":
            selected = [row for row in selected if row["case"] in COMMON_7_CASES]
        selected.sort(key=lambda row: row["case"])
        if not selected:
            continue

        result: dict[str, str | int] = {
            "record_type": "case_equal_summary",
            "scope": scope,
            "case": "",
            "method": METHOD_LABELS[method],
            "case_count": len(selected),
            "included_cases": ";".join(row["case"] for row in selected),
            "segment_count": sum(int(row["segment_count"]) for row in selected),
            "i2v_subject_valid_segments": sum(
                int(row["i2v_subject_valid_segments"]) for row in selected
            ),
        }
        for output_name, source_name in METRICS:
            values = [parse_number(row[source_name]) for row in selected]
            valid_values = [value for value in values if value is not None]
            result[output_name] = number_text(statistics.fmean(valid_values)) if valid_values else ""
        output.append(result)
    return output


def per_case(rows: list[dict[str, str]]) -> list[dict[str, str | int]]:
    output: list[dict[str, str | int]] = []
    case_rank = {case: index for index, case in enumerate(CASE_ORDER)}
    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    for row in sorted(rows, key=lambda item: (case_rank[item["case"]], method_rank[item["method"]])):
        result: dict[str, str | int] = {
            "record_type": "per_case",
            "scope": "all_cases_including_goblin",
            "case": row["case"],
            "method": METHOD_LABELS[row["method"]],
            "case_count": 1,
            "included_cases": row["case"],
            "segment_count": int(row["segment_count"]),
            "i2v_subject_valid_segments": int(row["i2v_subject_valid_segments"]),
        }
        for output_name, source_name in METRICS:
            value = parse_number(row[source_name])
            result[output_name] = number_text(value) if value is not None else ""
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.case_summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = summarize(rows, "all_available_cases_including_goblin")
    output_rows += summarize(rows, "common_7_cases_including_goblin")
    output_rows += per_case(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(args.output.resolve())


if __name__ == "__main__":
    main()
