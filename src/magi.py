import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from src.log import info, status, success

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "character"

EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def find_manga_pages(directory: Path) -> list[Path]:
    pages: list[tuple[int, Path]] = []
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manga_pages = find_manga_pages(TARGET_DIR)
    info(f"Found {len(manga_pages)} page(s): {[p.name for p in manga_pages]}")
    images = [read_image_as_np_array(p) for p in manga_pages]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with status(f"Loading MAGI model on {device}..."):
        model = AutoModel.from_pretrained(
            "ragavsachdeva/magi",
            trust_remote_code=True,
        ).to(device)
    success(f"MAGI model loaded ({device})")

    with status("Detecting + OCR..."), torch.no_grad():
        results = model.predict_detections_and_associations(images)
        text_bboxes = [x["texts"] for x in results]
        ocr_results = model.predict_ocr(images, text_bboxes)

    out_path = OUTPUT_DIR / "transcript.txt"
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (result, ocr) in enumerate(zip(results, ocr_results)):
            tmp_path = Path(tmp) / f"transcript_{i}.txt"
            model.generate_transcript_for_single_image(
                result, ocr, filename=str(tmp_path)
            )
            parts.append(tmp_path.read_text(encoding="utf-8"))

    header_line = " ### Transcript ###\n"
    stripped = [p.replace(header_line, "") for p in parts]
    out_path.write_text(header_line + "".join(stripped), encoding="utf-8")
    success(f"Transcript saved to {out_path}")


if __name__ == "__main__":
    main()
