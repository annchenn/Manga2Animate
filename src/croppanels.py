"""
Crop manga panels from one or more manga pages using the MAGI model.

Input : target/<target>/manga_page.*  and/or  manga_page1.*, manga_page2.*, ...
Output: target/<target>/manga_panels/panel1.png, panel2.png, ...  (sequential across all pages)

Panels are sorted in manga reading order: top-to-bottom rows,
right-to-left within each row.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.log import info, step, success, warn


BASE_DIR = Path(__file__).resolve().parent.parent


EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def find_manga_pages(directory: Path) -> list[Path]:
    """Return manga_page.* and manga_page1.*, manga_page2.*, ... sorted by page number."""
    pages: list[tuple[int | float, Path]] = []
    for p in directory.iterdir():
        if p.suffix.lower() not in EXTS:
            continue
        name = p.stem
        if name == "manga_page":
            pages.append((0, p))
        elif name.startswith("manga_page") and name[len("manga_page"):].isdigit():
            pages.append((int(name[len("manga_page"):]), p))
    if not pages:
        raise FileNotFoundError(f"No manga_page* image found in {directory}")
    pages.sort(key=lambda x: x[0])
    return [p for _, p in pages]


def read_image_as_np_array(image_path: Path) -> np.ndarray:
    with open(image_path, "rb") as f:
        image = Image.open(f).convert("L").convert("RGB")
        return np.array(image)


def sort_panels_reading_order(bboxes, image_height, debug=False):
    if not bboxes:
        return []

    # Step 1: compute center points
    items = []
    for b in bboxes:
        x1, y1, x2, y2 = b
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        items.append((b, cx, cy))

    # Step 2: sort by y-center
    items.sort(key=lambda x: x[2])

    rows = []
    row_tolerance = image_height * 0.06  # ⭐ stable threshold

    # Step 3: clustering into rows (stable version)
    for bbox, cx, cy in items:
        placed = False

        for row in rows:
            # row center
            row_cy = sum(r[2] for r in row) / len(row)

            if abs(cy - row_cy) <= row_tolerance:
                row.append((bbox, cx, cy))
                placed = True
                break

        if not placed:
            rows.append([(bbox, cx, cy)])

    # Step 4: sort rows top → bottom
    rows.sort(key=lambda r: sum(x[2] for x in r) / len(r))

    # Step 5: inside row sort right → left
    ordered = []
    for row in rows:
        row_sorted = sorted(row, key=lambda x: -x[1])
        ordered.extend([x[0] for x in row_sorted])

    return ordered


def main():
    parser = argparse.ArgumentParser(description="Crop manga panels via MAGI model")
    parser.add_argument("-t", "--target", required=True, help="Target name under target/")
    args = parser.parse_args()

    target_dir = BASE_DIR / "target" / args.target
    out_dir    = target_dir / "manga_panels"

    manga_pages = find_manga_pages(target_dir)
    info(f"Found {len(manga_pages)} page(s): {[p.name for p in manga_pages]}")
    info(f"Output: {out_dir}")

    img_arrays = [read_image_as_np_array(p) for p in manga_pages]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    step(f"Loading MAGI model on {device}...")
    model = AutoModel.from_pretrained(
        "ragavsachdeva/magi",
        trust_remote_code=True,
    ).to(device)
    success(f"MAGI model loaded ({device})")

    step("Detecting panels...")
    with torch.no_grad():
        results = model.predict_detections_and_associations(img_arrays)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel_idx = 1

    for page_num, (manga_path, img_array, result) in enumerate(
        zip(manga_pages, img_arrays, results), start=1
    ):
        raw_panels = result["panels"]
        info(f"Page {page_num} ({manga_path.name}): {len(raw_panels)} panel(s) detected")

        if not raw_panels:
            warn(f"  No panels detected on page {page_num} — skipping.")
            continue

        ordered = sort_panels_reading_order(
            [list(b) for b in raw_panels], img_array.shape[0]
        )

        src_image = Image.fromarray(img_array)
        for bbox in ordered:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cropped = src_image.crop((x1, y1, x2, y2))
            out_path = out_dir / f"panel{panel_idx}.png"
            cropped.save(out_path)
            info(f"  panel{panel_idx}.png  bbox=({x1},{y1},{x2},{y2})")
            panel_idx += 1

    total = panel_idx - 1
    if total == 0:
        warn("No panels saved.")
    else:
        success(f"Saved {total} panel(s) to {out_dir}")


if __name__ == "__main__":
    main()
