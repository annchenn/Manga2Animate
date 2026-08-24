"""
Crop manga panels from one or more manga pages using the MAGI model.

Input : target/<target>/manga_page.*  and/or  manga_page1.*, manga_page2.*, ...
Output: target/<target>/manga_panels/panel1.png, panel2.png, ...  (sequential across all pages)

Panels are sorted using MAGI's built-in reading-order algorithm
(directed-graph topological sort with cycle breaking).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from shapely.geometry import box
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
        image = Image.open(f).convert("RGB")
        return np.array(image)


def to_grayscale_rgb(img_array: np.ndarray) -> np.ndarray:
    gray = Image.fromarray(img_array).convert("L").convert("RGB")
    return np.array(gray)




def main():
    parser = argparse.ArgumentParser(description="Crop manga panels via MAGI model")
    parser.add_argument("-t", "--target", required=True, help="Target name under target/")
    args = parser.parse_args()

    target_dir = BASE_DIR / "target" / args.target
    out_dir    = target_dir / "manga_panels"

    manga_pages = find_manga_pages(target_dir)
    info(f"Found {len(manga_pages)} page(s): {[p.name for p in manga_pages]}")
    info(f"Output: {out_dir}")

    img_arrays  = [read_image_as_np_array(p) for p in manga_pages]
    gray_arrays = [to_grayscale_rgb(a) for a in img_arrays]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    step(f"Loading MAGI model on {device}...")
    model = AutoModel.from_pretrained(
        "ragavsachdeva/magi",
        trust_remote_code=True,
    ).to(device)
    success(f"MAGI model loaded ({device})")

    step("Detecting panels...")
    with torch.no_grad():
        results = model.predict_detections_and_associations(gray_arrays, panel_detection_threshold=0.2,)

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

        text_bboxes = result["texts"]
        sorted_panel_indices, _ = model.sort_panels_and_text_bboxes_in_reading_order(
            [raw_panels], [text_bboxes]
        )
        ordered = [raw_panels[i] for i in sorted_panel_indices[0]]

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
