#!/usr/bin/env python3
"""Convert character reference images to JPEG.

CIDS only globs for `.jpg` (`cids_evaluator.py:673`), so a `.png` reference is
invisible to it and the run dies with "Cant get ref char". This converts every
non-jpg reference image under the dataset to `NN.jpg`.

RGBA/LA/P images are flattened onto white — JPEG has no alpha channel.

The original is renamed to `<name>.<ext>.bak` rather than deleted, so it stops
being picked up by `dataset_load.py` (which does collect .png) without losing
the source file. Pass --delete to remove originals instead.

Examples
--------
    python src/normalize_ref_images.py                 # whole dataset
    python src/normalize_ref_images.py --story 03      # one story
    python src/normalize_ref_images.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
BENCH_DIR = SRC_DIR.parent

DEFAULT_DATASET_ROOT = BENCH_DIR / "data" / "dataset"
DATASET_NAME = "ViStory"

# Extensions dataset_load.py collects, minus the one CIDS actually wants.
CONVERTIBLE = {".png", ".jpeg", ".webp", ".bmp"}


def info(msg: str) -> None:
    print(f"\033[94m{msg}\033[0m")


def success(msg: str) -> None:
    print(f"\033[92m{msg}\033[0m")


def warn(msg: str) -> None:
    print(f"\033[93mWARN: {msg}\033[0m")


def next_free_index(char_dir: Path) -> int:
    """Lowest NN not already taken by an NN.jpg in this folder."""
    used = set()
    for p in char_dir.glob("*.jpg"):
        if p.stem.isdigit():
            used.add(int(p.stem))
    n = 0
    while n in used:
        n += 1
    return n


def convert_one(src: Path, dst: Path) -> str:
    im = Image.open(src)
    mode = im.mode
    if mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGB", im.size, (255, 255, 255))
        flat.paste(im, mask=im.split()[-1])
        out = flat
    else:
        out = im.convert("RGB")
    out.save(dst, "JPEG", quality=95)
    return f"{mode}, {im.size[0]}x{im.size[1]}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert character reference images to JPEG for CIDS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--story", default=None, help="Only this story id, e.g. 03")
    parser.add_argument("--delete", action="store_true",
                        help="Delete originals instead of renaming them to .bak")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_root) / DATASET_NAME
    if not dataset_dir.is_dir():
        warn(f"{dataset_dir} does not exist")
        return 1

    stories = [dataset_dir / args.story] if args.story else sorted(
        d for d in dataset_dir.iterdir() if d.is_dir() and d.name.isdigit()
    )

    converted = empty = ok = skipped_stories = 0
    for story_dir in stories:
        image_dir = story_dir / "image"
        if not image_dir.is_dir():
            continue

        char_dirs = sorted(d for d in image_dir.iterdir() if d.is_dir())
        # A story with no reference images at all is simply not set up for
        # evaluation yet; bench_run skips it. Only a partially-filled story is
        # worth warning about, since that is the case that breaks CIDS midway.
        if not any(any(d.iterdir()) for d in char_dirs):
            skipped_stories += 1
            continue

        for char_dir in char_dirs:
            todo = sorted(p for p in char_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in CONVERTIBLE)
            has_jpg = any(char_dir.glob("*.jpg"))

            if not todo:
                if has_jpg:
                    ok += 1
                else:
                    warn(f"{story_dir.name}/{char_dir.name}: no reference image at all "
                         f"-> CIDS will fail for this character")
                    empty += 1
                continue

            for src in todo:
                dst = char_dir / f"{next_free_index(char_dir):02d}.jpg"
                if args.dry_run:
                    info(f"  {story_dir.name}/{char_dir.name}: {src.name} -> {dst.name} (dry run)")
                    converted += 1
                    continue
                detail = convert_one(src, dst)
                if args.delete:
                    src.unlink()
                    action = "deleted original"
                else:
                    src.rename(src.with_suffix(src.suffix + ".bak"))
                    action = f"kept as {src.name}.bak"
                info(f"  {story_dir.name}/{char_dir.name}: {src.name} ({detail}) -> {dst.name}, {action}")
                converted += 1

    print()
    skip_note = f", {skipped_stories} story(ies) skipped (no reference images)" if skipped_stories else ""
    success(f"Converted {converted} image(s); {ok} character folder(s) already jpg-only{skip_note}.")
    if empty:
        warn(f"{empty} character folder(s) have no reference image — CIDS will fail on those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
