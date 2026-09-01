#!/usr/bin/env python3
"""Copy each story's ground-truth manga panels into the ViStoryBench dataset.

OCCM asks "does the generated shot show as many characters as the source panel does",
so the evaluator needs the panel next to the story it belongs to. The panels live with
the pipeline's inputs (`<target-root>/<target>/manga_panels/panelN.png`); this copies
them to where the evaluator looks:

    data/dataset/ViStory/<story_id>/manga_panels/panelN.png

`<story_id>` comes from the dataset's story_index.json, the same mapping the rest of
the benchmark uses. Stories with no panels are reported and skipped — the evaluator
falls back to the script's character count for those.

Examples
--------
    python src/link_manga_panels.py
    python src/link_manga_panels.py -t goblin -t mado
    python src/link_manga_panels.py --target-root ../../target
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
BENCH_DIR = SRC_DIR.parent
BASE_DIR = SRC_DIR.parents[2]

DATASET_NAME = "ViStory"
INDEX_FILE_NAME = "story_index.json"
PANEL_DIR_NAME = "manga_panels"
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--target", action="append", dest="targets",
                        help="target name (repeatable); default: every target in story_index.json")
    parser.add_argument("--target-root", type=Path, default=BASE_DIR / "target_test",
                        help="folder holding <target>/manga_panels/")
    parser.add_argument("--dataset-root", type=Path, default=BENCH_DIR / "data" / "dataset")
    args = parser.parse_args()

    dataset_dir = args.dataset_root / DATASET_NAME
    index_path = dataset_dir / INDEX_FILE_NAME
    if not index_path.exists():
        print(f"ERROR: {index_path} not found", file=sys.stderr)
        return 1
    story_index = json.loads(index_path.read_text(encoding="utf-8"))

    targets = args.targets or list(story_index)
    missing, done = [], 0
    for target in targets:
        if target not in story_index:
            print(f"WARN: {target} is not in {INDEX_FILE_NAME}, skipping")
            missing.append(target)
            continue
        src = args.target_root / target / PANEL_DIR_NAME
        panels = sorted(p for p in src.glob("*") if p.suffix.lower() in EXTS) if src.is_dir() else []
        if not panels:
            print(f"WARN: no panels under {src}")
            missing.append(target)
            continue

        dst = dataset_dir / story_index[target] / PANEL_DIR_NAME
        dst.mkdir(parents=True, exist_ok=True)
        for panel in panels:
            shutil.copy2(panel, dst / panel.name)
        print(f"{target} ({story_index[target]}): {len(panels)} panel(s) -> {dst}")
        done += 1

    if missing:
        print(f"no panels for: {', '.join(missing)}")
    print(f"done: {done}/{len(targets)} target(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
