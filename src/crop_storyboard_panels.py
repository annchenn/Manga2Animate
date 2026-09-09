"""
Crop panels from an AI-generated storyboard image (bestStoryboard.png) using MAGI,
with two fixes on top of raw MAGI output:

  1. Sort order: MAGI's built-in sort assumes manga (right-to-left) reading order.
     Storyboards use left-to-right, top-to-bottom order, so we use a simple
     geometric row-major sort instead.
  2. Border trim: MAGI's box regression is not pixel-exact and leaves a few px
     of the panel's black outline stroke, plus a sliver of page background,
     on each side of the crop. We trim both by scanning inward from each edge
     until we hit real content.

Input : <results-root>/<target>/<run timestamp>/iter<N>/bestStoryboard.png
        (latest run and latest iter by default; pick a run with --time)
Output: <results-root>/<target>/<run timestamp>/iter<N>/storyboard_panels/
            panel1.png, panel2.png, ...   top-left -> bottom-right order, border trimmed
            overview.png                  storyboard with numbered boxes (for visual check)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.log import info, step, success, warn
from src.runs import resolve_run_dir

BASE_DIR = Path(__file__).resolve().parent.parent

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

ROW_GAP_FRAC = 0.15        # y-gap (as a fraction of image height) that starts a new row
BLACK_THRESH = 30          # a row/col whose max pixel value is below this is treated as black gutter
FLAT_STD_THRESH = 15       # a row/col this uniform (any color) is treated as border bleed, not content
MAX_TRIM_FRAC = 0.08       # never trim more than this fraction of width/height per side


def find_best_storyboard(run_dir: Path, iteration: int | None) -> Path:
    """Return bestStoryboard.* from the given iter, or the latest iter that has one."""
    if iteration is not None:
        iter_dirs = [run_dir / f"iter{iteration}"]
    else:
        iter_dirs = sorted(
            (d for d in run_dir.glob("iter*") if d.is_dir()),
            key=lambda d: int(d.name[len("iter"):]),
            reverse=True,
        )
    for d in iter_dirs:
        for ext in EXTS:
            p = d / f"bestStoryboard{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"No bestStoryboard image found under {run_dir}")


def read_image_as_np_array(image_path: Path) -> np.ndarray:
    with open(image_path, "rb") as f:
        image = Image.open(f).convert("RGB")
        return np.array(image)


def to_grayscale_rgb(img_array: np.ndarray) -> np.ndarray:
    gray = Image.fromarray(img_array).convert("L").convert("RGB")
    return np.array(gray)


def sort_top_left_to_bottom_right(boxes, img_h: int):
    """Row-major sort: group boxes into rows by y-proximity, then left -> right within each row."""
    boxes = sorted(boxes, key=lambda b: b[1])  # by y1 first
    rows = []
    for box in boxes:
        placed = False
        for row in rows:
            if abs(box[1] - row[0][1]) < ROW_GAP_FRAC * img_h:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    ordered = []
    for row in rows:
        row.sort(key=lambda b: b[0])  # left -> right
        ordered.extend(row)
    return ordered


def group_into_rows(boxes, img_h: int):
    """Boxes grouped by y-proximity, each row sorted left -> right."""
    rows = []
    for box in sorted(boxes, key=lambda b: b[1]):
        for row in rows:
            if abs(box[1] - row[0][1]) < ROW_GAP_FRAC * img_h:
                row.append(box)
                break
        else:
            rows.append([box])
    for row in rows:
        row.sort(key=lambda b: b[0])
    return rows


def reconcile_to_grid(boxes, img_w: int, img_h: int, expected: int):
    """Force MAGI's boxes onto the storyboard's regular grid.

    Detection is not reliable enough to trust the count: on a 56-sheet batch it returned
    something other than six panels on 16 of them, in two ways. A panel whose content is
    mostly blank is missed entirely, and a panel with a strong internal edge is returned
    as two boxes. Either way the writer numbers whatever it got as panel1..panelN, so one
    missing panel silently shifts every later shot onto the wrong image -- the failure is
    invisible until the benchmark scores shot 4 against shot 5's prompt.

    Where the storyboard is a regular grid the geometry is the ground truth: boxes are
    grouped into rows and columns, several boxes landing in one cell are merged, and a
    cell with no box at all is synthesised from its row's and column's extent.

    Not every storyboard is a grid, though -- some put three panels in one column beside
    a single wide one -- so this returns (None, reason) when the boxes cannot be fitted,
    and the caller keeps the raw detections rather than inventing a layout. It is only
    ever consulted when the count is already wrong, so a correct irregular layout is
    never touched.

    Returns the grid in reading order plus human-readable repairs, empty when nothing
    needed fixing.
    """
    rows = group_into_rows(boxes, img_h)
    if not rows:
        return [], []
    n_rows = len(rows)
    if n_rows == 0 or expected % n_rows:
        return None, [f"cannot fit {len(boxes)} box(es) into {n_rows} row(s) for {expected} panels"]
    n_cols = expected // n_rows

    # Column edges come from the row that MAGI split most finely, falling back to an even
    # division when no row shows all of them.
    widest = max(rows, key=len)
    if len(widest) >= n_cols:
        centres = sorted((b[0] + b[2]) / 2 for b in widest)
        # Collapse to n_cols centres by taking the largest gaps as the column separators.
        while len(centres) > n_cols:
            gaps = [(centres[i + 1] - centres[i], i) for i in range(len(centres) - 1)]
            _g, i = min(gaps)
            centres[i:i + 2] = [(centres[i] + centres[i + 1]) / 2]
        edges = [0] + [(centres[i] + centres[i + 1]) / 2 for i in range(n_cols - 1)] + [img_w]
    else:
        edges = [img_w * i / n_cols for i in range(n_cols + 1)]

    repairs, grid = [], []
    for r, row in enumerate(rows):
        y1 = min(b[1] for b in row)
        y2 = max(b[3] for b in row)
        cells = [[] for _ in range(n_cols)]
        for b in row:
            centre = (b[0] + b[2]) / 2
            c = min(range(n_cols), key=lambda i: abs(centre - (edges[i] + edges[i + 1]) / 2))
            cells[c].append(b)
        for c, found in enumerate(cells):
            if len(found) == 1:
                grid.append(found[0])
            elif len(found) > 1:
                grid.append((min(b[0] for b in found), min(b[1] for b in found),
                             max(b[2] for b in found), max(b[3] for b in found)))
                repairs.append(f"row {r + 1} col {c + 1}: merged {len(found)} boxes into one panel")
            else:
                grid.append((int(edges[c]), y1, int(edges[c + 1]), y2))
                repairs.append(f"row {r + 1} col {c + 1}: no box detected, filled from the grid")
    return grid, repairs


def is_border_line(line: np.ndarray) -> bool:
    """True if this row/col is either near-pure-black (gutter) or flat/uniform (any-color bleed,
    e.g. page background or an adjacent panel's edge color showing through anti-aliasing)."""
    if line.max() < BLACK_THRESH:
        return True
    return bool(line.std() < FLAT_STD_THRESH)


def trim_black_border(image: Image.Image) -> Image.Image:
    """Shrink inward from each edge while the row/col is border (black gutter or white page bg)."""
    arr = np.array(image.convert("L"))
    h, w = arr.shape
    max_trim_h = int(h * MAX_TRIM_FRAC)
    max_trim_w = int(w * MAX_TRIM_FRAC)

    top = 0
    while top < max_trim_h and is_border_line(arr[top, :]):
        top += 1
    bottom = h
    while (h - bottom) < max_trim_h and is_border_line(arr[bottom - 1, :]):
        bottom -= 1

    # Scan left/right using only the rows that survived the top/bottom trim, so a
    # busy region elsewhere in the full column doesn't mask a real border near the edge.
    vert_slice = arr[top:bottom, :]
    left = 0
    while left < max_trim_w and is_border_line(vert_slice[:, left]):
        left += 1
    right = w
    while (w - right) < max_trim_w and is_border_line(vert_slice[:, right - 1]):
        right -= 1

    if bottom <= top or right <= left:
        return image  # safety fallback, shouldn't happen with sane thresholds
    return image.crop((left, top, right, bottom))


def main():
    parser = argparse.ArgumentParser(description="Crop storyboard panels via MAGI model")
    parser.add_argument("-t", "--target", required=True, help="Target name under <results-root>/")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=BASE_DIR / "results_test",
        help="Results root directory (default: results/; e.g. results_test/)",
    )
    parser.add_argument(
        "--time",
        default=None,
        help="Pipeline run timestamp under <results-root>/<target>/ (default: the latest run)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Iteration to use (default: latest iter that has a bestStoryboard)",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=6,
        help="Panels the storyboard should have (default: 6). A detection that misses or "
             "splits one is reconciled against the grid; 0 disables the check.",
    )
    args = parser.parse_args()

    target_dir = args.results_root / args.target
    run_dir = resolve_run_dir(target_dir, args.time)
    storyboard_path = find_best_storyboard(run_dir, args.iter)
    out_dir = storyboard_path.parent / "storyboard_panels"

    info(f"Run: {run_dir.name}")
    info(f"Storyboard: {storyboard_path}")
    info(f"Output: {out_dir}")

    img_array = read_image_as_np_array(storyboard_path)
    gray_array = to_grayscale_rgb(img_array)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    step(f"Loading MAGI model on {device}...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).to(device)
    success(f"MAGI model loaded ({device})")

    step("Detecting panels...")
    with torch.no_grad():
        result = model.predict_detections_and_associations([gray_array])[0]

    raw_panels = result["panels"]
    info(f"{len(raw_panels)} panel(s) detected")
    if not raw_panels:
        warn("No panels detected.")
        return

    boxes = [tuple(int(v) for v in bbox) for bbox in raw_panels]
    h, w = img_array.shape[0], img_array.shape[1]
    ordered = sort_top_left_to_bottom_right(boxes, h)

    # Detection alone is not trustworthy enough to number the output by: a blank-ish panel
    # is missed and a panel with a strong internal edge comes back as two, and either one
    # shifts every later panel onto the wrong shot without any sign that it happened.
    if args.expect and len(ordered) != args.expect:
        warn(f"{len(ordered)} panel(s) detected but {args.expect} expected — "
             f"reconciling against the storyboard grid")
        repaired, repairs = reconcile_to_grid(ordered, w, h, args.expect)
        if repaired is None or len(repaired) != args.expect:
            for note in repairs:
                warn(f"  {note}")
            warn("Could not recover the grid; writing the raw detections instead. "
                 "Check overview.png before trusting the panel numbering.")
        else:
            for note in repairs:
                info(f"  {note}")
            ordered = repaired
            success(f"Recovered {args.expect} panels from the grid")

    out_dir.mkdir(parents=True, exist_ok=True)
    src_image = Image.fromarray(img_array)
    overview = src_image.copy()
    draw = ImageDraw.Draw(overview)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    for idx, bbox in enumerate(ordered, start=1):
        x1, y1, x2, y2 = bbox
        cropped = trim_black_border(src_image.crop((x1, y1, x2, y2)))
        out_path = out_dir / f"panel{idx}.png"
        cropped.save(out_path)
        info(f"  panel{idx}.png  raw_bbox=({x1},{y1},{x2},{y2})  trimmed_size={cropped.size}")

        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1 + 6, y1 + 6), str(idx), fill="red", font=font)

    overview_path = out_dir / "overview.png"
    overview.save(overview_path)
    success(f"Saved {len(ordered)} panel(s) to {out_dir}")


if __name__ == "__main__":
    main()
