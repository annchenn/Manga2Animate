#!/usr/bin/env python3
"""Extract one shot frame per AniMaker clip video, for ViStoryBench scoring.

AniMaker (https://github.com/HITsz-TMG/Anim-Director, arXiv 2506.10540) is the
baseline we compared ourselves against. Unlike our own storyboard video — one clip
holding all shots back to back, see ../../../src/crop_video_frames.py, which needs
cut detection to split it — AniMaker's own pipeline already renders one independent
video file per shot (its Stage 5 "chosen" clip, named `scene_<scene_pos>_clip_<id>.mp4`).
So there is no cut detection to do here: we just grab a single frame out of each clip.

Input : a folder of AniMaker's chosen clip videos, `scene_<n>_clip_<n>.mp4`, copied
        straight out of its `Pipeline/res/<story>/videos/chosen/` — nothing else from
        AniMaker's pipeline output is needed (no script.json/pipeline.json/imgs/...).
        Shot order is derived from the filenames themselves: sorted by
        (scene_pos, clip_id), then numbered 1..K in that order.
Writes: <outputs-root>/<method>/<mode>/<language>/<timestamp>/<story>/shots/
            shot_1.png ... shot_K.png   ViStoryBench shot order

Frame choice defaults to the middle frame of each clip (`--frame mid`; also supports
`first`/`last`) — the middle frame is the single frame most representative of what
the clip actually shows, avoiding the fade-in/fade-out right at the clip boundaries.

Examples
--------
    python benchmark/vistorybench/src/export_animaker_shots.py \\
        --videos-dir /path/to/animaker_chosen_clips --story 03

    python benchmark/vistorybench/src/export_animaker_shots.py \\
        --videos-dir /path/to/animaker_chosen_clips --story 03 --frame first
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

SRC_DIR = Path(__file__).resolve().parent
BENCH_DIR = SRC_DIR.parent

DEFAULT_OUTPUTS_ROOT = BENCH_DIR / "data" / "outputs"

CLIP_NAME_RE = re.compile(r"^scene_(\d+)_clip_(\d+)\.mp4$")


def info(msg: str) -> None:
    print(f"\033[94m{msg}\033[0m")


def success(msg: str) -> None:
    print(f"\033[92m{msg}\033[0m")


def warn(msg: str) -> None:
    print(f"\033[93mWARN: {msg}\033[0m")


def error(msg: str) -> None:
    print(f"\033[91mERROR: {msg}\033[0m")


def enumerate_clips(videos_dir: Path) -> list[tuple[int, int, int, Path]]:
    """Return [(shot_num, scene_pos, clip_id, path), ...], ordered by (scene_pos, clip_id).

    Files not matching `scene_<n>_clip_<n>.mp4` are skipped (with a warning) rather
    than aborting the whole batch.
    """
    found = []
    skipped = []
    for p in sorted(Path(videos_dir).iterdir()):
        if not p.is_file():
            continue
        m = CLIP_NAME_RE.match(p.name)
        if not m:
            skipped.append(p.name)
            continue
        found.append((int(m.group(1)), int(m.group(2)), p))
    if skipped:
        warn(f"skipped {len(skipped)} file(s) not matching scene_<n>_clip_<n>.mp4: {skipped}")
    found.sort(key=lambda t: (t[0], t[1]))
    return [(i, scene_pos, clip_id, p) for i, (scene_pos, clip_id, p) in enumerate(found, start=1)]


def extract_frame(video_path: Path, dst_png: Path, which: str = "mid") -> None:
    """Save a single frame from video_path as dst_png. which: mid|first|last.

    Ported as-is from AniMaker's own export_vistory_shots.py::_extract_frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if which == "first":
        idx = 0
    elif which == "last":
        idx = max(total - 1, 0)
    else:
        idx = max(total // 2, 0)

    frame = None
    if total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
    if frame is None:  # seek can silently fail on some codecs — fall back to sequential read
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frame = f
            if i >= idx:
                break
            i += 1
    cap.release()
    if frame is None:
        raise RuntimeError(f"no readable frame in {video_path}")
    cv2.imwrite(str(dst_png), frame)


def extract_frames(video_path: Path, dst_dir: Path, count: int) -> int:
    """
    Save `count` evenly spaced frames from one clip into dst_dir as frame_00.png ...

    One clip is one shot here, so uniform sampling over the whole clip is the shot's
    10-frame sample: ViStoryBench scores each frame and averages them. Returns how
    many frames were written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # Sequential read, because seeking is unreliable on some of these codecs and we
    # want every frame anyway.
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"no readable frame in {video_path} (reported {total} frames)")

    if count <= 1:
        picks = [len(frames) // 2]
    else:
        step = (len(frames) - 1) / (count - 1)
        picks = [min(len(frames) - 1, round(i * step)) for i in range(count)]

    dst_dir.mkdir(parents=True, exist_ok=True)
    for f_i, idx in enumerate(picks):
        cv2.imwrite(str(dst_dir / f"frame_{f_i:02d}.png"), frames[idx])
    return len(picks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos-dir", type=Path, required=True,
                        help="folder containing AniMaker's chosen clip videos (scene_<n>_clip_<n>.mp4)")
    parser.add_argument("--story", required=True, help="ViStoryBench story id, e.g. 03")
    parser.add_argument("--frame", choices=["mid", "first", "last"], default="mid",
                        help="which frame to take from each clip when --frames-per-shot 1")
    parser.add_argument("--frames-per-shot", type=int, default=10,
                        help="frames sampled per clip for scoring; 1 writes the flat "
                             "shot_N.png layout using --frame instead")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--method", default="anim_video", help="baseline name under data/outputs/")
    parser.add_argument("--mode", default="base")
    parser.add_argument("--language", default="en")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="YYYYMMDD_HHMMSS; reuse one to overwrite a previous extraction")
    args = parser.parse_args()

    if not args.videos_dir.is_dir():
        error(f"--videos-dir does not exist or is not a directory: {args.videos_dir}")
        sys.exit(1)

    clips = enumerate_clips(args.videos_dir)
    if not clips:
        error(f"no scene_<n>_clip_<n>.mp4 files found in {args.videos_dir}")
        sys.exit(1)

    shots_dir = (args.outputs_root / args.method / args.mode / args.language
                 / args.timestamp / args.story / "shots")
    shots_dir.mkdir(parents=True, exist_ok=True)
    info(f"{len(clips)} clip(s) -> {shots_dir}")

    failed = []
    for shot_num, scene_pos, clip_id, video_path in clips:
        try:
            if args.frames_per_shot > 1:
                dst = shots_dir / f"shot_{shot_num}"
                if dst.exists():
                    shutil.rmtree(dst)
                n = extract_frames(video_path, dst, args.frames_per_shot)
                print(f"  shot_{shot_num}/ ({n} frames)  <-  {video_path.name}  (scene_{scene_pos}/clip_{clip_id})")
            else:
                dst = shots_dir / f"shot_{shot_num}.png"
                extract_frame(video_path, dst, which=args.frame)
                print(f"  shot_{shot_num}.png  <-  {video_path.name}  (scene_{scene_pos}/clip_{clip_id})")
        except Exception as e:
            failed.append(shot_num)
            warn(f"shot {shot_num} ({video_path.name}): {e}")

    ok = len(clips) - len(failed)
    if failed:
        warn(f"{ok}/{len(clips)} shots extracted; failed: {failed}")
    else:
        success(f"{ok}/{len(clips)} shots extracted -> {shots_dir}")

    info("next: copy this tree to wherever run_bench.sh actually reads data/outputs/ from, then:")
    print(f"  ./run_bench.sh --method {args.method} --mode {args.mode}")


if __name__ == "__main__":
    main()
