#!/usr/bin/env python3
"""Normalise every baseline's native output layout into the ViStoryBench outputs tree.

Each baseline ships its results in its own shape — `panel1.png`, `img_0.png`,
`frame_01.png`, one mp4 per shot, one mp4 for the whole story — and the benchmark reads
exactly one shape:

    data/outputs/<method>/base/en/<timestamp>/<story_id>/shots/shot_N.png       (storyboard)
    data/outputs/<method>/base/en/<timestamp>/<story_id>/shots/shot_N/frame_XX.png  (video)

Storyboard baselines are copied straight across, renumbered to 1..N so a zero-indexed
source lines up with `story.json`. Video baselines are sampled at 10 frames per shot: a
per-shot-clip baseline samples each clip, and a single-video baseline goes through the
same cut detection our own pipeline uses before sampling inside each detected shot.

    python src/import_baseline_outputs.py --list
    python src/import_baseline_outputs.py                   # every baseline
    python src/import_baseline_outputs.py -b storydiffusion -b storyiter
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

SRC_DIR = Path(__file__).resolve().parent
BENCH_DIR = SRC_DIR.parent
BASE_DIR = SRC_DIR.parents[2]

sys.path.insert(0, str(BASE_DIR))
from src.crop_video_frames import (  # noqa: E402
    MIN_SHOT_SEC, _hist, _small, find_cuts, read_video, sample_segment,
)

DEFAULT_EVAL_ROOT = BASE_DIR / "evaluation_results"
DEFAULT_OUTPUTS_ROOT = BENCH_DIR / "data" / "outputs"
DATASET_NAME = "ViStory"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
NUM_SHOTS = 6
FRAMES_PER_SHOT = 10

# (method, track, source subdir, layout, story-dir glob)
#   flat_images  : the story directory holds one image per shot
#   per_shot_clip: the story directory holds one mp4 per shot
#   single_video : one mp4 per story, all shots concatenated
BASELINES = [
    ("ours_scene",     "storyboard", None,                          "ours_panels",   "*"),
    ("anim_scene",     "storyboard", "animaker/anim_scene",         "flat_images",   "*"),
    ("storydiffusion", "storyboard", "storydiffusion",              "flat_images",   "*"),
    ("storyiter",      "storyboard", "StoryIter",                   "flat_images",   "results_xl10_*"),
    ("anystory",       "storyboard", "AnyStory(0817meeting)/output", "flat_images",  "*"),
    ("anim_video",     "video",      "animaker/source_video",       "per_shot_clip", "*"),
    ("vimax",          "video",      "Vimax(0811meeting)",          "single_video",  "*_final_video.mp4"),
    ("storymem",       "video",      "StoryMem(0811meeting)",       "single_video",  "*.mp4"),
]

# Source folders spell the targets their own way; story_index.json is the authority.
ALIASES = {
    "lovehina": "LoveHina",
    "dualjustice1": "DualJustice1",
    "dualjustice2": "DualJustice2",
    "dualjustice": None,   # ambiguous: could be either DualJustice story, so never guess
}


def info(msg): print(f"\033[94m{msg}\033[0m")
def step(msg): print(f"\033[1;96m>> {msg}\033[0m")
def success(msg): print(f"\033[92m{msg}\033[0m")
def warn(msg): print(f"\033[93mWARN: {msg}\033[0m")


def resolve_target(name: str, story_index: dict) -> str | None:
    """Source folder name -> story id, or None when it maps to nothing we evaluate."""
    stem = re.sub(r"\.(mp4|mov|avi)$", "", name)   # extension first: the decorations below
    stem = re.sub(r"^results_xl10_", "", stem)      # are anchored to the end of the stem
    stem = re.sub(r"_final_video$", "", stem)
    if stem in story_index:
        return story_index[stem]
    alias = ALIASES.get(stem.lower(), ...)
    if alias is ...:
        for target, sid in story_index.items():          # case-insensitive fallback
            if target.lower() == stem.lower():
                return sid
        return None
    return story_index.get(alias) if alias else None


def sorted_shot_images(story_dir: Path) -> list[Path]:
    """
    Images in shot order, by the trailing number in each filename.

    Sources disagree on where numbering starts (`img_0.png` vs `panel1.png`), so the
    order matters and the absolute value does not — the caller renumbers to 1..N.
    """
    numbered = []
    for p in story_dir.iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.search(r"(\d+)$", p.stem.strip())
        if m:                                   # skips contact sheets like grid.png
            numbered.append((int(m.group(1)), p))
    return [p for _n, p in sorted(numbered)]


def find_ours_panels(results_root: Path, target: str) -> Path | None:
    """
    Newest `storyboard_panels/` for one of our targets.

    `crop_storyboard_panels.py` writes them beside the storyboard it cropped, which sits
    at `<target>/iter<N>/` on older runs and `<target>/<run timestamp>/iter<N>/` on newer
    ones. Sorting the found directories by modification time picks the latest crop under
    either layout without having to special-case them.
    """
    found = sorted((results_root / target).glob("**/storyboard_panels"),
                   key=lambda d: d.stat().st_mtime)
    return found[-1] if found else None


def import_flat_images(story_dir: Path, dst: Path) -> int:
    imgs = sorted_shot_images(story_dir)
    if not imgs:
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(imgs, start=1):
        shutil.copy2(src, dst / f"shot_{i}{src.suffix.lower()}")
    return len(imgs)


def sample_clip(video: Path, dst_dir: Path, count: int) -> int:
    """`count` evenly spaced frames from one clip (one clip == one shot)."""
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return 0
    picks = ([len(frames) // 2] if count <= 1 else
             [min(len(frames) - 1, round(i * (len(frames) - 1) / (count - 1))) for i in range(count)])
    dst_dir.mkdir(parents=True, exist_ok=True)
    for i, idx in enumerate(picks):
        cv2.imwrite(str(dst_dir / f"frame_{i:02d}.png"), frames[idx])
    return len(picks)


def import_per_shot_clips(story_dir: Path, dst: Path, count: int) -> int:
    """One mp4 per shot; shot order comes from (scene, clip) in the filenames."""
    clips = []
    for p in story_dir.glob("*.mp4"):
        m = re.search(r"scene_(\d+)_clip_(\d+)", p.stem)
        clips.append(((int(m.group(1)), int(m.group(2))) if m else (10**6, p.stem), p))
    if not clips:
        return 0
    for i, (_key, clip) in enumerate(sorted(clips, key=lambda kv: kv[0]), start=1):
        sample_clip(clip, dst / f"shot_{i}", count)
    return len(clips)


def import_single_video(video: Path, dst: Path, num_shots: int, count: int) -> int:
    """
    One mp4 holding every shot end to end.

    Same treatment as our own storyboard videos: detect the cuts first, then sample
    inside each detected segment. Sampling the whole file uniformly would straddle cuts
    and hand the same shot to two different frame slots.
    """
    frames, fps = read_video(video)
    if not frames:
        return 0
    smalls = [_small(f) for f in frames]
    hists = [_hist(s) for s in smalls]
    bounds, confident = find_cuts(hists, num_shots, min_gap=max(1, int(MIN_SHOT_SEC * fps)))
    info(f"    {len(frames)} frames @ {fps:.1f}fps ({len(frames)/fps:.1f}s), "
         f"{confident} hard cut(s) of {num_shots - 1}")
    for n in range(1, num_shots + 1):
        shot_dir = dst / f"shot_{n}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        for i, idx in enumerate(sample_segment(bounds[n - 1], bounds[n], count)):
            cv2.imwrite(str(shot_dir / f"frame_{i:02d}.png"), frames[idx])
    return num_shots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-b", "--baseline", action="append", dest="baselines",
                        help="method name (repeatable); default: all")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=BENCH_DIR / "data" / "dataset")
    parser.add_argument("--results-root", type=Path, default=BASE_DIR / "results_test",
                        help="where our own pipeline results live (for ours_scene)")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="output timestamp folder; reuse one to overwrite")
    parser.add_argument("--frames-per-shot", type=int, default=FRAMES_PER_SHOT)
    parser.add_argument("--num-shots", type=int, default=NUM_SHOTS)
    parser.add_argument("--list", action="store_true", help="show the registry and exit")
    args = parser.parse_args()

    if args.list:
        for method, track, sub, layout, glob in BASELINES:
            print(f"  {method:16s} {track:11s} {layout:14s} {sub or '(results_test)'}")
        return 0

    story_index = json.loads(
        (args.dataset_root / DATASET_NAME / "story_index.json").read_text(encoding="utf-8"))
    wanted = set(args.baselines) if args.baselines else None
    total_ok = 0

    for method, track, sub, layout, glob_pat in BASELINES:
        if wanted and method not in wanted:
            continue
        if layout == "ours_panels":
            entries = []
            for target in story_index:
                panels = find_ours_panels(args.results_root, target)
                if panels:
                    entries.append((target, panels))
                else:
                    warn(f"  {target}: no storyboard_panels/ yet "
                         f"(run ../../src/crop_storyboard_panels.py -t {target})")
            step(f"{method} ({track}) <- {args.results_root}/<target>/**/storyboard_panels")
        else:
            src_root = args.eval_root / sub
            if not src_root.is_dir():
                warn(f"{method}: {src_root} not found, skipping")
                continue
            step(f"{method} ({track}) <- {sub}")
            entries = [(e.name, e) for e in sorted(src_root.glob(glob_pat))]

        dst_root = args.outputs_root / method / "base" / "en" / args.timestamp
        done, skipped = 0, []
        for name, entry in entries:
            sid = resolve_target(name, story_index)
            if sid is None:
                skipped.append(name)
                continue
            dst = dst_root / sid / "shots"
            if dst.exists():
                shutil.rmtree(dst)

            if layout in ("flat_images", "ours_panels"):
                n = import_flat_images(entry, dst)
                detail = f"{n} shot image(s)"
            elif layout == "per_shot_clip":
                n = import_per_shot_clips(entry, dst, args.frames_per_shot)
                detail = f"{n} shot(s) x {args.frames_per_shot} frames"
            else:
                n = import_single_video(entry, dst, args.num_shots, args.frames_per_shot)
                detail = f"{n} shot(s) x {args.frames_per_shot} frames"

            if n:
                info(f"  {name} -> {sid}: {detail}")
                done += 1
            else:
                warn(f"  {entry.name} -> {sid}: nothing imported")
        if skipped:
            warn(f"  unmapped, skipped: {', '.join(skipped)}")
        success(f"  {done} story(ies) -> {dst_root}")
        total_ok += done

    print()
    success(f"imported {total_ok} story folder(s); timestamp {args.timestamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
