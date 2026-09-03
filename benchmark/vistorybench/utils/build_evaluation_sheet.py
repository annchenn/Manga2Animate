#!/usr/bin/env python3
"""Build the paper-style comparison table: one row per method, one sheet per track.

Storyboard and video go to separate sheets because they are separate tables in the
paper. Every baseline we intend to compare against gets a row whether or not it has
been run, so the table is its final shape from the start and an unrun baseline reads
as a dash rather than a gap.

The metric columns carry a coloured band naming the benchmark they come from. Today
everything is ViStoryBench; when VBench++ (or anything else) is added, give it its own
entry in METRIC_SOURCES and its columns get their own colour, so it stays obvious at a
glance which numbers came from which benchmark.

Numbers are averaged over the stories that ran; per-story numbers stay in
data/bench_results/<method>/<mode>/en/<ts>/metrics/*/story_results.json.

    python utils/build_evaluation_sheet.py -o /tmp/table.xlsx --markdown
    python utils/build_evaluation_sheet.py -o /tmp/table.csv     # one flat sheet

Writing .xlsx needs openpyxl; the vistorybench env has it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BENCH_DIR / "evaluation_table.xlsx"

STORIES = ["elf2", "elf3", "goblin", "mado", "LoveHina", "DualJustice1", "DualJustice2", "kick"]

# Model each method generates with. Ours comes from the pipeline's .env
# (IMAGE_MODEL / VIDEO_GENERATION_MODEL); the baselines are filled in as we confirm them.
MODELS = {
    ("Storyboard", "Ours"): "gemini-3.1-flash-image",
    ("Video", "Ours"): "seedance-2.0",
}
# Only the video table carries a Model column.
BLOCK_HAS_MODEL = {"Storyboard": False, "Video": True}

# The TBD tab: how much of each (track, method) is still outstanding. Combinations we
# have decided not to run at all are marked here rather than inferred — an image-only
# baseline has no video track to score, and a video-only one has no storyboard track.
NOT_NEEDED = {
    ("Storyboard", "Vimax"),
    ("Storyboard", "StoryMem"),
    ("Video", "StoryDiffusion"),
    ("Video", "StoryIter"),
    ("Video", "AnyStory"),
}
NOT_NEEDED_LABEL = "don't need"

# Rows of the table, per block. `None` for the output folder means "not run yet".
BLOCKS = [
    ("Storyboard", [
        ("Ours",           "ours_scene",     "base"),
        ("AniMaker",       "anim_scene",     "base"),
        ("Vimax",          None,             None),
        ("StoryDiffusion", "storydiffusion", "base"),
        ("StoryMem",       None,             None),
        ("StoryIter",      "storyiter",      "base"),
        ("AnyStory",       "anystory",       "base"),
    ]),
    ("Video", [
        ("Ours",           "ours_video", "base"),
        ("AniMaker",       "anim_video", "base"),
        ("Vimax",          "vimax",      "base"),
        ("StoryDiffusion", None,         None),
        ("StoryMem",       "storymem",   "base"),
        ("StoryIter",      None,         None),
        ("AnyStory",       None,         None),
    ]),
]

# Benchmark each column group comes from -> the colour its band gets. Add a row here
# when a new benchmark's metrics join the table.
METRIC_SOURCES = [
    ("ViStoryBench", "CSD↑ CIDS↑ PA↑ CM↑", "DCE9F5"),
]

# (group, sub-header, metric file, mean key, std key or None, decimals)
COLUMNS = [
    ("CSD↑",  "Cross", "csd",          "cross_csd",               None,                          3),
    ("CSD↑",  "Self",  "csd",          "self_csd",                None,                          3),
    ("CIDS↑", "Cross", "cids",         "cids_cross_mean",         None,                          3),
    ("CIDS↑", "Self",  "cids",         "cids_self_mean",          None,                          3),
    ("PA↑",   "Scene", "prompt_align", "scene",                   "scene_std",                   2),
    ("PA↑",   "Shot",  "prompt_align", "camera",                  "camera_std",                  2),
    ("PA↑",   "CI",    "prompt_align", "character_action",        "character_action_std",        2),
    ("PA↑",   "IA",    "cids",         "single_character_action", "single_character_action_std", 2),
    ("PA↑",   "Avg.",  None,           None,                      None,                          2),
    ("CM↑",   "",      "cids",         "occm",                    None,                          1),
]
PA_AVG_SOURCES = [("prompt_align", "scene"), ("prompt_align", "camera"),
                  ("prompt_align", "character_action"), ("cids", "single_character_action")]
PA_AVG_STDS = [("prompt_align", "scene_std"), ("prompt_align", "camera_std"),
               ("prompt_align", "character_action_std"), ("cids", "single_character_action_std")]

DASH = "—"
WIN_FILL = "FFF2CC"   # the best value in each metric column

# Per-story rows for the Detail sheet.
DETAIL_COLUMNS = [
    ("CSD Cross",  "csd",          "cross_csd",               None,                          4),
    ("CSD Self",   "csd",          "self_csd",                None,                          4),
    ("CIDS Cross", "cids",         "cids_cross_mean",         None,                          4),
    ("CIDS Self",  "cids",         "cids_self_mean",          None,                          4),
    ("PA Scene",   "prompt_align", "scene",                   "scene_std",                   2),
    ("PA Shot",    "prompt_align", "camera",                  "camera_std",                  2),
    ("PA CI",      "prompt_align", "character_action",        "character_action_std",        2),
    ("PA IA",      "cids",         "single_character_action", "single_character_action_std", 2),
    ("CM",         "cids",         "occm",                    None,                          2),
    # Not one of the four metrics we report, but the CIDS evaluator computes it anyway,
    # so it costs nothing to carry it through to the per-story sheet.
    ("Copy-Paste", "cids",         "copy_paste_score",        None,                          4),
]


def is_usable(run_dir: Path) -> bool:
    """
    True when this run is both finished and produced by the current evaluators.

    Two ways a run must be rejected:

    - **Unfinished.** `summary.json` is written by `compute_and_save_summary()` at the
      very end of a run, so a run still in progress does not have one. Without this
      check the newest (running) directory wins and the table silently mixes stories
      that have been scored with stories that have not, showing dashes for the rest.
    - **Stale.** Pre-SigLIP / pre-manga-panel-OCCM / pre-repeated-PA runs sort alongside
      the new ones but their numbers mean something different. Current results carry a
      `scoring` block; older ones do not.

    The `scoring` block is looked for in any metric the run computed, not in CIDS
    specifically: a PA-only backfill pass writes no CIDS at all, and demanding CIDS here
    would drop it from `resolve_runs` and leave the PA column empty even though the run
    finished cleanly.
    """
    if not (run_dir / "summary.json").exists():
        return False
    for metric in ("cids", "prompt_align"):
        path = run_dir / "metrics" / metric / "story_results.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(isinstance(o, dict) and isinstance(o.get("scoring"), dict) for o in data.values()):
            return True
    return False


def outputs_timestamp(run_dir: Path) -> str | None:
    """Which generation run this evaluation scored, from metadata.json."""
    try:
        meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (meta.get("run") or {}).get("outputs_timestamp")


def resolve_runs(results_root: Path, method: str, mode: str) -> dict:
    """
    metric -> the run its numbers should come from.

    Metrics get split across runs in practice: a fast sweep computes CSD/CIDS/OCCM with
    the VLM switched off, and PA arrives later from a separate pass. Taking only the
    newest run would throw the PA away, so each metric falls back to the newest earlier
    run that has it.

    The fallback is restricted to runs that scored the *same* generation outputs. PA
    measured against a different set of images is not the same measurement, even for the
    same method, so those numbers are left out rather than quietly mixed in.
    """
    base = results_root / method / mode / "en"
    if not base.is_dir():
        return {}
    runs = [d for d in sorted(base.iterdir(), key=lambda d: d.name, reverse=True)
            if d.is_dir() and is_usable(d)]
    if not runs:
        return {}

    primary = runs[0]
    want = outputs_timestamp(primary)
    resolved = {}
    for metric in ("csd", "cids", "prompt_align"):
        for run in runs:
            if run is not primary and outputs_timestamp(run) != want:
                continue
            if (run / "metrics" / metric / "story_results.json").exists():
                resolved[metric] = run
                if run is not primary:
                    print(f"note: {method}/{mode} {metric} from {run.name} "
                          f"(newest run {primary.name} did not compute it)", file=sys.stderr)
                break
    resolved["_primary"] = primary
    return resolved


def latest_run(results_root: Path, method: str, mode: str) -> Path | None:
    base = results_root / method / mode / "en"
    if not base.is_dir():
        return None
    runs = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    for run in runs:
        if is_usable(run):
            if run is not runs[0]:
                print(f"note: {method}/{mode} using {run.name}; {runs[0].name} is unfinished "
                      f"or predates the current metrics", file=sys.stderr)
            return run
    if runs:
        print(f"skipping {method}/{mode}: no finished run in the current metric format",
              file=sys.stderr)
    return None


def load_story_metrics(run_dir: Path, metric: str) -> dict:
    path = run_dir / "metrics" / metric / "story_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(sid): o["metrics"] for sid, o in data.items()
            if isinstance(o, dict) and isinstance(o.get("metrics"), dict)}


def mean_over_stories(per_metric, story_ids, metric, key):
    """Mean of one metric key across the stories that produced it, or None."""
    if metric is None or key is None:
        return None
    vals = [(per_metric.get(metric, {}).get(sid) or {}).get(key) for sid in story_ids]
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def cell(mean, std, decimals):
    """Plain number, matching the paper's table; a dash when the run does not exist.

    `std` is accepted and ignored: the repeated-scoring spread lives in each run's
    story_results.json, and putting `±x.xx` in every PA cell doubles the table's width
    for a number that is 0.00 on most rows.
    """
    return DASH if mean is None else f"{mean:.{decimals}f}"


def build_block(results_root: Path, story_index: dict, methods, block_name):
    """
    Rows, their numeric values, and provenance notes for one track.

    Methods marked NOT_NEEDED for this track are dropped rather than shown as a row of
    dashes: an image-only baseline has no video to score and vice versa, so its absence
    from the video table is not a gap anyone needs to see. The TBD sheet is where that
    decision stays visible.
    """
    story_ids = [story_index[s] for s in STORIES if s in story_index]
    rows, values, notes = [], [], []

    for display, method_dir, mode_dir in methods:
        if (block_name, display) in NOT_NEEDED:
            continue
        runs = resolve_runs(results_root, method_dir, mode_dir) if method_dir else {}
        run_dir = runs.get("_primary")
        per_metric = {m: load_story_metrics(runs[m], m) for m in ("csd", "cids", "prompt_align")
                      if m in runs}

        row = [display]
        if BLOCK_HAS_MODEL.get(block_name):
            row.append(MODELS.get((block_name, display), DASH))
        value_row = []
        for _group, _sub, metric, key, std_key, decimals in COLUMNS:
            if metric is None:  # PA Avg., derived from the four PA columns
                parts = [mean_over_stories(per_metric, story_ids, m, k) for m, k in PA_AVG_SOURCES]
                parts = [v for v in parts if v is not None]
                stds = [mean_over_stories(per_metric, story_ids, m, k) for m, k in PA_AVG_STDS]
                stds = [v for v in stds if v is not None]
                mean = sum(parts) / len(parts) if parts else None
                std = sum(stds) / len(stds) if stds else None
            else:
                mean = mean_over_stories(per_metric, story_ids, metric, key)
                std = mean_over_stories(per_metric, story_ids, metric, std_key) if std_key else None
            row.append(cell(mean, std, decimals))
            value_row.append(mean)
        rows.append(row)
        values.append(value_row)

        if run_dir:
            covered = sorted(set(per_metric.get("cids", {})))
            names = [s for s in STORIES if story_index.get(s) in covered]
            notes.append(f"{display}: {len(names)} stories ({', '.join(names)}) "
                         f"— {method_dir}/{mode_dir}/{run_dir.name}")
    return rows, values, notes


def build_tbd(results_root: Path, story_index: dict):
    """How many of the eight stories each (track, method) has scored."""
    story_ids = [story_index[s] for s in STORIES if s in story_index]
    total = len(STORIES)
    header = ["", "storyboard", "video"]
    by_method = {}
    order = []
    for block_name, methods in BLOCKS:
        for display, method_dir, mode_dir in methods:
            if display not in by_method:
                by_method[display] = {}
                order.append(display)
            if (block_name, display) in NOT_NEEDED:
                by_method[display][block_name] = NOT_NEEDED_LABEL
                continue
            runs = resolve_runs(results_root, method_dir, mode_dir) if method_dir else {}
            done = 0
            if runs.get("cids"):
                scored = load_story_metrics(runs["cids"], "cids")
                done = sum(1 for sid in story_ids if sid in scored)
            by_method[display][block_name] = f"{done}/{total}" if done else ""
    rows = [[m, by_method[m].get("Storyboard", ""), by_method[m].get("Video", "")] for m in order]
    return header, rows


def build_detail(results_root: Path, story_index: dict):
    """Every method x story x metric, one row per (track, method, story)."""
    header = ["Track", "Method", "Story", "Story ID"]
    for name, _m, _k, std_key, _d in DETAIL_COLUMNS:
        header.append(name)
        if std_key:
            header.append(f"{name} std")
    header.append("Run")

    rows = []
    for block_name, methods in BLOCKS:
        for display, method_dir, mode_dir in methods:
            if (block_name, display) in NOT_NEEDED:   # same filter as the two main tables
                continue
            runs = resolve_runs(results_root, method_dir, mode_dir) if method_dir else {}
            run_dir = runs.get("_primary")
            per_metric = {m: load_story_metrics(runs[m], m) for m in ("csd", "cids", "prompt_align")
                          if m in runs}
            run_label = f"{method_dir}/{mode_dir}/{run_dir.name}" if run_dir else ""
            for story in STORIES:
                sid = story_index.get(story, "")
                row = [block_name, display, story, sid]
                for _name, metric, key, std_key, decimals in DETAIL_COLUMNS:
                    v = (per_metric.get(metric, {}).get(sid) or {}).get(key)
                    row.append(round(float(v), decimals) if isinstance(v, (int, float)) else DASH)
                    if std_key:
                        sv = (per_metric.get(metric, {}).get(sid) or {}).get(std_key)
                        row.append(round(float(sv), 4) if isinstance(sv, (int, float)) else DASH)
                row.append(run_label if sid in (per_metric.get("cids") or {}) else "")
                rows.append(row)
    return header, rows


FOOTNOTES = [
    "PA is the mean of 5 VLM scorings per item; ± is the spread across them.",
    "CM (OCCM) compares the character count GroundingDINO finds in the generated shot "
    "against the ground-truth manga panel, prompt 'character', threshold 0.25.",
    "CIDS identity is scored with SigLIP.",
    f"Judge: gemini-2.5-flash (temperature 1.0, top_p 1.0). {DASH} = not run yet.",
]
VIDEO_FOOTNOTE = "Video scores average 10 frames sampled per shot."


def write_xlsx(path: Path, results_root: Path, story_index: dict):
    """Storyboard, Video, and Detail sheets.

    Storyboard/Video are the paper-style tables: two header rows at the very top, the
    metric group merged across its sub-columns, and the best value in each metric column
    given a background so the winner is readable at a glance. Every metric here is
    higher-is-better, so "best" is always the maximum. Detail carries the per-story
    numbers behind those averages.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    thin = Side(style="thin", color="808080")
    hair = Side(style="thin", color="C0C0C0")

    colour_of_group = {}
    for group, *_ in COLUMNS:
        for _name, groups, colour in METRIC_SOURCES:
            if group in groups.split():
                colour_of_group[group] = colour

    for block_name, methods in BLOCKS:
        ws = wb.create_sheet(block_name)
        rows, values, _notes = build_block(results_root, story_index, methods, block_name)
        fixed_labels = ("Method", "Model") if BLOCK_HAS_MODEL.get(block_name) else ("Method",)
        FIXED = len(fixed_labels)

        for i, label in enumerate(fixed_labels):
            c = ws.cell(row=1, column=1 + i, value=label)
            c.font = Font(bold=True)
            c.alignment = Alignment(horizontal="left", vertical="center")
            ws.merge_cells(start_row=1, start_column=1 + i, end_row=2, end_column=1 + i)
            ws.cell(row=2, column=1 + i).border = Border(bottom=thin)

        for i, (group, sub, *_rest) in enumerate(COLUMNS):
            g = ws.cell(row=1, column=FIXED + 1 + i, value=group)
            g.font = Font(bold=True)
            g.alignment = Alignment(horizontal="center")
            if group in colour_of_group:
                g.fill = PatternFill("solid", fgColor=colour_of_group[group])
            sc = ws.cell(row=2, column=FIXED + 1 + i, value=sub)
            sc.font = Font(bold=True)
            sc.alignment = Alignment(horizontal="center")
            sc.border = Border(bottom=thin)
        i = 0
        while i < len(COLUMNS):
            j = i
            while j + 1 < len(COLUMNS) and COLUMNS[j + 1][0] == COLUMNS[i][0]:
                j += 1
            ws.merge_cells(start_row=1, start_column=FIXED + 1 + i,
                           end_row=1 if j > i else 2, end_column=FIXED + 1 + j)
            i = j + 1

        # Winner per metric column. Ties get highlighted together rather than picking one.
        best_rows = []
        for col in range(len(COLUMNS)):
            present = [(r, v[col]) for r, v in enumerate(values) if v[col] is not None]
            if len(present) < 2:
                best_rows.append(set())
                continue
            top = max(v for _r, v in present)
            best_rows.append({r for r, v in present if v == top})

        for r, row in enumerate(rows, start=3):
            for c, value in enumerate(row, start=1):
                ref = ws.cell(row=r, column=c, value=value)
                ref.alignment = Alignment(horizontal="left" if c <= FIXED else "center")
                if c > FIXED:
                    ref.border = Border(left=hair)
                    if (r - 3) in best_rows[c - FIXED - 1]:
                        ref.fill = PatternFill("solid", fgColor=WIN_FILL)
                        ref.font = Font(bold=True)

        ws.column_dimensions["A"].width = 16
        if FIXED > 1:
            ws.column_dimensions["B"].width = 22
        for i in range(len(COLUMNS)):
            ws.column_dimensions[get_column_letter(FIXED + 1 + i)].width = 8
        ws.freeze_panes = ws.cell(row=3, column=FIXED + 1)

    # Detail: every method x story, the numbers the averages above are built from.
    ws = wb.create_sheet("Detail")
    header, rows = build_detail(results_root, story_index)
    for c, label in enumerate(header, start=1):
        cell_ref = ws.cell(row=1, column=c, value=label)
        cell_ref.font = Font(bold=True)
        cell_ref.alignment = Alignment(horizontal="center")
        cell_ref.border = Border(bottom=thin)
    for r, row in enumerate(rows, start=2):
        for c, value in enumerate(row, start=1):
            ref = ws.cell(row=r, column=c, value=value)
            ref.alignment = Alignment(horizontal="left" if c <= 4 else "center")
    ws.column_dimensions["A"].width = 11
    ws.column_dimensions["B"].width = 15
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 9
    for c in range(5, len(header) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 11
    ws.column_dimensions[get_column_letter(len(header))].width = 34
    ws.freeze_panes = "E2"

    # TBD: which (track, method) combinations still need running.
    ws = wb.create_sheet("TBD")
    header, rows = build_tbd(results_root, story_index)
    for c, label in enumerate(header, start=1):
        ref = ws.cell(row=1, column=c, value=label)
        ref.font = Font(bold=True)
        ref.alignment = Alignment(horizontal="center")
        ref.border = Border(bottom=thin)
    for r, row in enumerate(rows, start=2):
        for c, value in enumerate(row, start=1):
            ref = ws.cell(row=r, column=c, value=value)
            ref.alignment = Alignment(horizontal="left" if c == 1 else "center")
            if c == 1:
                ref.font = Font(bold=(row[0] == "Ours"))
            elif value == NOT_NEEDED_LABEL:
                ref.font = Font(color="909090", italic=True)
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 13
    ws.column_dimensions["C"].width = 13

    wb.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f".xlsx (default: {DEFAULT_OUTPUT.name}) gives one sheet per track "
                             f"with colours; .csv gives a single flat sheet")
    parser.add_argument("--tsv", type=Path, default=None, metavar="DIR",
                        help="also write one .tsv per sheet into DIR, for pasting a single "
                             "tab into an existing spreadsheet")
    parser.add_argument("--results-root", type=Path, default=BENCH_DIR / "data" / "bench_results")
    parser.add_argument("--dataset-root", type=Path, default=BENCH_DIR / "data" / "dataset")
    parser.add_argument("--markdown", action="store_true", help="also print the tables as markdown")
    parser.add_argument("--no-workbook", action="store_true",
                        help="write only the --tsv files, skipping the .xlsx/.csv workbook")
    args = parser.parse_args()
    if args.no_workbook and not args.tsv:
        parser.error("--no-workbook needs --tsv, otherwise nothing would be written")

    index_path = args.dataset_root / "ViStory" / "story_index.json"
    story_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    if not args.no_workbook:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".xlsx":
            write_xlsx(args.output, args.results_root, story_index)
        else:
            with args.output.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for block_name, methods in BLOCKS:
                    fixed = ["Method", "Model"] if BLOCK_HAS_MODEL.get(block_name) else ["Method"]
                    group_header = fixed + [c[0] for c in COLUMNS]
                    sub_header = [""] * len(fixed) + [c[1] for c in COLUMNS]
                    rows, _values, _notes = build_block(args.results_root, story_index, methods, block_name)
                    w.writerow([block_name])
                    w.writerow(group_header)
                    w.writerow(sub_header)
                    w.writerows(rows)
                    w.writerow([])

    if args.tsv:
        args.tsv.mkdir(parents=True, exist_ok=True)
        for block_name, methods in BLOCKS:
            fixed = ["Method", "Model"] if BLOCK_HAS_MODEL.get(block_name) else ["Method"]
            group_header = fixed + [c[0] for c in COLUMNS]
            sub_header = [""] * len(fixed) + [c[1] for c in COLUMNS]
            rows, _values, _notes = build_block(args.results_root, story_index, methods, block_name)
            out = args.tsv / f"{block_name}.tsv"
            out.write_text("\n".join("\t".join(str(c) for c in r)
                                     for r in [group_header, sub_header] + rows) + "\n",
                           encoding="utf-8")
            print(f"  {out}", file=sys.stderr)
        header, rows = build_detail(args.results_root, story_index)
        out = args.tsv / "Detail.tsv"
        out.write_text("\n".join("\t".join(str(c) for c in r) for r in [header] + rows) + "\n",
                       encoding="utf-8")
        print(f"  {out}", file=sys.stderr)
        header, rows = build_tbd(args.results_root, story_index)
        out = args.tsv / "TBD.tsv"
        out.write_text("\n".join("\t".join(str(c) for c in r) for r in [header] + rows) + "\n",
                       encoding="utf-8")
        print(f"  {out}", file=sys.stderr)

    if args.markdown:
        for block_name, methods in BLOCKS:
            fixed = ["Method", "Model"] if BLOCK_HAS_MODEL.get(block_name) else ["Method"]
            group_header = fixed + [c[0] for c in COLUMNS]
            sub_header = [""] * len(fixed) + [c[1] for c in COLUMNS]
            rows, _values, _notes = build_block(args.results_root, story_index, methods, block_name)
            widths = [max(len(str(r[i])) for r in [group_header, sub_header] + rows)
                      for i in range(len(group_header))]
            def line(cells):
                return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
            print(f"\n### {block_name}")
            print(line(group_header))
            print(line(sub_header))
            print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
            for r in rows:
                print(line(r))

    if not args.no_workbook:
        print(f"\n{args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
