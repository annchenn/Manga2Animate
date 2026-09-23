"""
Extract one representative frame per shot from a generated video, so the video can
be scored by ViStoryBench's image metrics (prompt_align, cids, csd, aesthetic).

The videos are generated from a 3x2 storyboard, so each clip contains 6 shots laid
end to end. Uniform time sampling would land frames mid-cut and can take two frames
from the same shot, so instead we:

  1. Detect cuts from the HSV-histogram distance between consecutive frames, taking
     the strongest peaks under a minimum-shot-length constraint. This gives exactly
     `--num-frames` segments, one per storyboard panel.
  2. Pick one frame inside each segment, scoring candidates on sharpness (to skip
     motion blur near a cut) and on distance from the previously picked frame (so
     consecutive outputs really do differ in shot angle).

Input : <results-root>/<target>/video/<model>/<newest run_*>/best_video.mp4 (reference-
        conditioned mode) or final_story.mp4 (first-frame mode), i.e. what
        src/create_video.py writes; or any video given with --video
Output: <video's folder>/video_frames/
            shot_1.png ... shot_6.png   representative frame per shot, ViStoryBench order
            shot_1/frame_00.png ...     the frames sampled inside that shot
            overview.png                3x2 contact sheet (for visual check)
        <outputs-root>/<method>/<mode>/<language>/<timestamp>/<story_id>/shots/
            shot_1/frame_00.png ...     copy for the benchmark; --no-bench-copy skips it

With --frames-per-shot 1 the benchmark copy is the flat `shot_1.png ... shot_6.png`
layout instead. ViStoryBench scores every frame in a `shot_<n>/` directory and averages
them, so the default of 10 is the "sample 10 frames per shot" evaluation protocol; the
representative frame stays the cut-aware pick described above and is what the contact
sheet shows.

The frames live next to the video they came from; the benchmark folder gets only the
shot files, because ViStoryBench's loader parses every image name in `shots/` as
`..._<index>.<ext>` and an `overview.png` there makes bench_run.py crash.

`<story_id>` comes from the dataset's story_index.json, so the copy lands where
bench_run.py already looks for it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.log import info, step, success, warn

BASE_DIR = Path(__file__).resolve().parent.parent

# create_video.py's final output: best_video.mp4 on the reference-conditioned route,
# final_story.mp4 (the per-panel clips concatenated) on the first-frame route.
VIDEO_NAMES = ("best_video.mp4", "final_story.mp4")
FRAMES_DIRNAME = "video_frames"
INDEX_FILE_NAME = "story_index.json"
DATASET_NAME = "ViStory"

NUM_FRAMES = 6                          # 3x2 storyboard
FRAMES_PER_SHOT = 10                    # frames sampled inside each shot for scoring
MIN_SHOT_SEC = 1.0                      # a real shot is never shorter than this
CUT_REL_FRAC = 0.25                     # a cut must be this fraction of the clip's strongest cut
CUT_ABS_FLOOR = 0.12                    # ...and at least this far apart, so near-static clips stay quiet
EDGE_FRAC = 0.15                        # ignore this much of each segment's head/tail (transition frames)
CANDIDATE_STRIDE = 2                    # subsample candidates inside a segment
HIST_BINS = (16, 16)                    # H,S bins for the colour part of the descriptor
GRAY_BINS = 64                          # intensity bins; carries the signal on black-and-white clips
SMALL_W = 160                           # frames are downscaled to this width before scoring
DIFF_WEIGHT = 0.5                       # weight of "differs from previous pick" vs sharpness


def _small(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return cv2.resize(frame, (SMALL_W, max(1, round(h * SMALL_W / w))), interpolation=cv2.INTER_AREA)


def _hist(frame_small: np.ndarray) -> np.ndarray:
    """
    Frame signature: H-S histogram concatenated with a grayscale intensity histogram.

    The colour half carries most of the signal on normal clips; the intensity half
    is what keeps black-and-white clips from scoring flat everywhere.
    """
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    color = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    color = cv2.normalize(color, color).flatten()
    inten = cv2.calcHist([gray], [0], None, [GRAY_BINS], [0, 256])
    inten = cv2.normalize(inten, inten).flatten()
    return np.concatenate([color, inten])


def _sharpness(frame_small: np.ndarray) -> float:
    """Variance of the Laplacian — low on motion-blurred and dissolve frames."""
    return float(cv2.Laplacian(cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def read_video(video_path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames, fps


def find_cuts(hists: list[np.ndarray], num_frames: int, min_gap: int) -> tuple[list[int], int]:
    """
    Return (num_frames+1 segment boundaries, number of confident cuts found).

    Two stages, because taking the top num_frames-1 peaks unconditionally is what
    put two frames inside one continuous pan: a slow pan's largest wobble ranks
    high when the clip simply has fewer hard cuts than panels.

      1. Accept only peaks that clear an adaptive threshold — a fraction of the
         clip's own strongest jump. Absolute thresholds do not transfer between
         colour and black-and-white clips, whose distance scales differ ~2x.
      2. If that leaves fewer segments than panels, repeatedly split the *longest*
         remaining segment at its own point of maximum change. Splitting the long
         segment keeps the two resulting frames as far apart as the clip allows.
    """
    n = len(hists)
    dists = np.array([
        cv2.compareHist(hists[i].astype(np.float32), hists[i + 1].astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
        for i in range(n - 1)
    ])
    threshold = max(CUT_ABS_FLOOR, CUT_REL_FRAC * float(dists.max()))

    cuts: list[int] = []
    for idx in np.argsort(dists)[::-1]:
        if dists[idx] < threshold or len(cuts) == num_frames - 1:
            break
        cut = int(idx) + 1
        if cut < min_gap or cut > n - min_gap:
            continue
        if any(abs(cut - c) < min_gap for c in cuts):
            continue
        cuts.append(cut)

    confident = len(cuts)
    while len(cuts) < num_frames - 1:
        bounds = [0] + sorted(cuts) + [n]
        _, i = max((bounds[k + 1] - bounds[k], k) for k in range(len(bounds) - 1))
        lo, hi = bounds[i] + min_gap, bounds[i + 1] - min_gap
        cuts.append(lo + int(np.argmax(dists[lo:hi])) if hi > lo else (bounds[i] + bounds[i + 1]) // 2)

    return [0] + sorted(cuts) + [n], confident


def pick_frame(
    seg_start: int,
    seg_end: int,
    smalls: list[np.ndarray],
    hists: list[np.ndarray],
    prev_hist: np.ndarray | None,
) -> int:
    """Pick the frame in [seg_start, seg_end) that is sharp and unlike the previous pick."""
    span = seg_end - seg_start
    pad = int(span * EDGE_FRAC)
    lo, hi = seg_start + pad, max(seg_start + pad + 1, seg_end - pad)
    candidates = list(range(lo, hi, CANDIDATE_STRIDE)) or [(seg_start + seg_end) // 2]

    sharp = np.array([_sharpness(smalls[i]) for i in candidates])
    sharp = sharp / sharp.max() if sharp.max() > 0 else np.zeros_like(sharp)

    if prev_hist is None:
        diff = np.zeros_like(sharp)
    else:
        diff = np.array([
            cv2.compareHist(prev_hist.astype(np.float32), hists[i].astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            for i in candidates
        ])
        diff = diff / diff.max() if diff.max() > 0 else diff

    return candidates[int(np.argmax((1 - DIFF_WEIGHT) * sharp + DIFF_WEIGHT * diff))]


def sample_segment(seg_start: int, seg_end: int, count: int) -> list[int]:
    """
    `count` evenly spaced frame indices inside one shot's segment.

    The same EDGE_FRAC head/tail trim as pick_frame is applied first, so the samples
    stay clear of the transition frames on either side of a cut. Uniform spacing is
    deliberate: the average over these frames is meant to represent the whole shot,
    not its sharpest moment.
    """
    span = seg_end - seg_start
    pad = int(span * EDGE_FRAC)
    lo, hi = seg_start + pad, max(seg_start + pad + 1, seg_end - pad)
    if count <= 1:
        return [(lo + hi) // 2]
    step = (hi - 1 - lo) / (count - 1)
    return [min(hi - 1, lo + round(i * step)) for i in range(count)]


def save_overview(frames: list[np.ndarray], picks: list[int], fps: float, out_path: Path) -> None:
    """3x2 contact sheet in the same reading order as the source storyboard."""
    cols, rows = 3, 2
    thumbs = []
    for n, idx in enumerate(picks, start=1):
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((480, 270))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        except Exception:
            font = ImageFont.load_default()
        label = f"{n}  t={idx / fps:.2f}s"
        draw.rectangle([0, 0, 210, 40], fill="black")
        draw.text((8, 6), label, fill="white", font=font)
        thumbs.append(img)

    w, h = thumbs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    for i, img in enumerate(thumbs[: cols * rows]):
        sheet.paste(img, ((i % cols) * w, (i // cols) * h))
    sheet.save(out_path)


def latest_run_video(target_dir: Path, model: str) -> Path:
    """The final video of the newest create_video.py run that finished."""
    model_dir = target_dir / "video" / model
    runs = sorted((d for d in model_dir.glob("run_*") if d.is_dir()), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No run_* directory under {model_dir}")
    for run in runs:
        for name in VIDEO_NAMES:
            if (run / name).exists():
                return run / name
    raise FileNotFoundError(f"No {' / '.join(VIDEO_NAMES)} in any run under {model_dir}")


def process_target(
    target: str,
    video_path: Path,
    frames_dir: Path,
    bench_dir: Path | None,
    num_frames: int,
    frames_per_shot: int = FRAMES_PER_SHOT,
) -> None:
    step(f"{target}: {video_path}")

    frames, fps = read_video(video_path)
    smalls = [_small(f) for f in frames]
    hists = [_hist(s) for s in smalls]
    info(f"  {len(frames)} frames @ {fps:.2f} fps ({len(frames) / fps:.1f}s)")

    bounds, confident = find_cuts(hists, num_frames, min_gap=max(1, int(MIN_SHOT_SEC * fps)))
    if confident < num_frames - 1:
        info(f"  {confident} hard cut(s); {num_frames - 1 - confident} boundary(ies) from subdividing long shots")

    picks: list[int] = []
    prev_hist = None
    for i in range(num_frames):
        idx = pick_frame(bounds[i], bounds[i + 1], smalls, hists, prev_hist)
        picks.append(idx)
        prev_hist = hists[idx]

    frames_dir.mkdir(parents=True, exist_ok=True)
    shot_files = []
    for n, idx in enumerate(picks, start=1):
        shot_file = frames_dir / f"shot_{n}.png"
        cv2.imwrite(str(shot_file), frames[idx])
        shot_files.append(shot_file)
        seg = f"[{bounds[n - 1] / fps:5.2f}s-{bounds[n] / fps:5.2f}s]"
        info(f"  shot_{n}.png  {seg}  frame={idx}  t={idx / fps:.2f}s")

    # Frames sampled across each shot; ViStoryBench averages its scores over these.
    shot_frame_dirs = []
    if frames_per_shot > 1:
        for n in range(1, num_frames + 1):
            shot_dir = frames_dir / f"shot_{n}"
            shot_dir.mkdir(parents=True, exist_ok=True)
            for f_i, idx in enumerate(sample_segment(bounds[n - 1], bounds[n], frames_per_shot)):
                cv2.imwrite(str(shot_dir / f"frame_{f_i:02d}.png"), frames[idx])
            shot_frame_dirs.append(shot_dir)
        info(f"  sampled {frames_per_shot} frames per shot into shot_1/ ... shot_{num_frames}/")

    save_overview(frames, picks, fps, frames_dir / "overview.png")
    success(f"  wrote {num_frames} frames to {frames_dir}")

    if bench_dir is not None:
        bench_dir.mkdir(parents=True, exist_ok=True)
        if shot_frame_dirs:
            for shot_dir in shot_frame_dirs:
                dst = bench_dir / shot_dir.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(shot_dir, dst)
        else:
            for shot_file in shot_files:
                shutil.copy2(shot_file, bench_dir / shot_file.name)
        info(f"  copied shots to {bench_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--target", action="append", dest="targets",
                        help="target name (repeatable); default: every target in story_index.json")
    parser.add_argument("--results-root", type=Path, default=BASE_DIR / "results")
    parser.add_argument("-m", "--model", default="seedance",
                        help="video model folder under <results-root>/<target>/video/ (the -m given "
                             "to create_video.py); the newest run with a final video is used")
    parser.add_argument("--video", type=Path, default=None,
                        help="score this video file instead of looking one up; needs exactly one -t")
    parser.add_argument("--dataset-root", type=Path, default=BASE_DIR / "mangaeval/vistorybench/data/dataset")
    parser.add_argument("--outputs-root", type=Path, default=BASE_DIR / "mangaeval/vistorybench/data/outputs")
    parser.add_argument("--method", default="manga2animate_video")
    parser.add_argument("--mode", default="base")
    parser.add_argument("--language", default="en")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help="output timestamp folder; reuse one to overwrite a previous extraction")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--frames-per-shot", type=int, default=FRAMES_PER_SHOT,
                        help="frames sampled inside each shot for scoring; "
                             "1 writes the flat shot_N.png layout instead")
    parser.add_argument("--no-bench-copy", action="store_true",
                        help="only write the video_frames/ folder next to the video, "
                             "do not copy shots into the ViStoryBench outputs tree")
    args = parser.parse_args()
    if args.video is not None:
        if not args.targets or len(args.targets) != 1:
            parser.error("--video needs exactly one -t, the story it belongs to")
        if not args.video.is_file():
            parser.error(f"--video: {args.video} not found")

    index_path = args.dataset_root / DATASET_NAME / INDEX_FILE_NAME
    story_index = json.loads(index_path.read_text())
    targets = args.targets or list(story_index)

    source = args.video or f"{args.results_root}/<target>/video/{args.model}/<newest run_*>"
    info(f"video   <- {source}")
    info(f"frames  -> <video's folder>/{FRAMES_DIRNAME}")
    if args.no_bench_copy:
        info("outputs -> (skipped, --no-bench-copy)")
    else:
        info(f"outputs -> {args.outputs_root}/{args.method}/{args.mode}/{args.language}/{args.timestamp}")

    failed = []
    for target in targets:
        if target not in story_index:
            warn(f"{target}: not in {index_path.name}, skipping")
            failed.append(target)
            continue
        try:
            video_path = args.video or latest_run_video(args.results_root / target, args.model)
        except FileNotFoundError as e:
            warn(f"{target}: {e}")
            failed.append(target)
            continue

        frames_dir = video_path.parent / FRAMES_DIRNAME
        bench_dir = None if args.no_bench_copy else (
            args.outputs_root / args.method / args.mode / args.language
            / args.timestamp / story_index[target] / "shots")
        process_target(target, video_path, frames_dir, bench_dir, args.num_frames,
                       args.frames_per_shot)

    if failed:
        warn(f"skipped: {', '.join(failed)}")
    success(f"done: {len(targets) - len(failed)}/{len(targets)} target(s)")


if __name__ == "__main__":
    main()
