"""CLI: score images/videos with no-reference anime aesthetic metrics.

Standalone tool -- not wired into VBench or vistorybench. Either framework
can call this from its own environment (subprocess) and read back the CSV;
this script does not assume anything about their output-tree layouts beyond
"a path is either an image, a video, or a directory of them".

Usage:
    python score.py <path> [<path> ...] [options]

See README.md for setup, output format, and license notes (aesthetic_shadow_v2's
weights are CC-BY-NC-4.0 / non-commercial).
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from metrics import METRICS

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm"}


def warn(msg: str) -> None:
    print(f"  ! {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"  · {msg}")


def discover(paths: list[str], recursive: bool) -> tuple[list[Path], list[Path]]:
    """Split input paths into (images, videos), expanding directories."""
    images: list[Path] = []
    videos: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
            elif p.suffix.lower() in VIDEO_EXTS:
                videos.append(p)
            else:
                warn(f"skipping {p}: not a recognized image/video extension")
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for f in it:
                if not f.is_file():
                    continue
                if f.suffix.lower() in IMAGE_EXTS:
                    images.append(f)
                elif f.suffix.lower() in VIDEO_EXTS:
                    videos.append(f)
        else:
            warn(f"skipping {p}: no such file or directory")
    return sorted(images), sorted(videos)


def sample_video_frames(video_path: Path, n: int) -> list["object"]:
    """n evenly spaced frames as PIL Images. No shot-cut detection -- this is
    a generic scorer, not story/shot-aware."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise RuntimeError(f"no frames reported for {video_path}")
        count = max(1, min(n, total))
        indices = [round(i * (total - 1) / max(1, count - 1)) for i in range(count)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(Image.fromarray(frame[:, :, ::-1]))  # BGR -> RGB
        if not frames:
            raise RuntimeError(f"could not decode any frame from {video_path}")
        return frames
    finally:
        cap.release()


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="image file(s), video file(s), and/or directories of them")
    parser.add_argument("--metrics", default=",".join(METRICS),
                         help=f"comma-separated metric names to run (default: all -- {','.join(METRICS)})")
    parser.add_argument("--out", type=Path, default=None,
                         help="output CSV path (default: ./scores_<timestamp>.csv)")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--frames-per-video", type=int, default=8,
                         help="frames sampled per video, evenly spaced (default: 8)")
    parser.add_argument("--device", default=None, help="cuda|cpu (default: auto-detect)")
    args = parser.parse_args()

    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metric_names if m not in METRICS]
    if unknown:
        parser.error(f"unknown metric(s): {', '.join(unknown)}. Available: {', '.join(METRICS)}")

    device = resolve_device(args.device)
    out_path = args.out or Path(f"scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    images, videos = discover(args.paths, args.recursive)
    if not images and not videos:
        parser.error("no images or videos found in the given path(s)")
    info(f"found {len(images)} image(s), {len(videos)} video(s)")

    # {content_type: {path: [PIL.Image, ...]}} -- built once, reused across metrics
    # so a video's frames aren't re-decoded per metric.
    video_frames: dict[Path, list] = {}
    for v in videos:
        try:
            video_frames[v] = sample_video_frames(v, args.frames_per_video)
        except Exception as e:
            warn(f"{v}: {e}")

    rows: list[dict] = []
    summary: dict[tuple[str, str], list[float]] = {}  # (path, metric) -> scores

    for metric_name in metric_names:
        info(f"loading {metric_name} ({device})...")
        metric = METRICS[metric_name]()
        try:
            metric.load(device)
        except Exception as e:
            warn(f"failed to load {metric_name}: {e}. Skipping this metric.")
            continue

        for img_path in images:
            try:
                img = Image.open(img_path)
                score = metric.score(img)
            except Exception as e:
                warn(f"{metric_name} failed on {img_path}: {e}")
                continue
            rows.append({"path": str(img_path), "content_type": "image", "frame_index": "",
                         "metric": metric_name, "score": score})
            summary.setdefault((str(img_path), metric_name), []).append(score)

        for v_path, frames in video_frames.items():
            for i, frame in enumerate(frames):
                try:
                    score = metric.score(frame)
                except Exception as e:
                    warn(f"{metric_name} failed on {v_path} frame {i}: {e}")
                    continue
                rows.append({"path": str(v_path), "content_type": "video_frame", "frame_index": i,
                             "metric": metric_name, "score": score})
                summary.setdefault((str(v_path), metric_name), []).append(score)

    if not rows:
        warn("no scores were produced")
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "content_type", "frame_index", "metric", "score"])
        writer.writeheader()
        writer.writerows(rows)
    info(f"wrote {len(rows)} row(s) to {out_path}")

    print()
    print("Summary (mean per path, per metric):")
    for (path, metric_name), scores in sorted(summary.items()):
        mean = sum(scores) / len(scores)
        print(f"  {metric_name:20s} {mean:8.4f}  ({len(scores)} sample(s))  {path}")


if __name__ == "__main__":
    main()
