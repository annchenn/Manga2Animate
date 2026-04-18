from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel

# --- Configuration ---
BASE_DIR = Path(__file__).parent
TARGET_DIR = BASE_DIR / "target"
OUTPUT_DIR = BASE_DIR / "results" / "character"


def find_image(directory: Path, stem: str) -> Path:
    matches = list(directory.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"No image found for '{stem}' in {directory}")
    return matches[0]


def read_image_as_np_array(image_path: Path) -> np.ndarray:
    with open(image_path, "rb") as f:
        image = Image.open(f).convert("L").convert("RGB")
        return np.array(image)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manga_path = find_image(TARGET_DIR, "manga_page")
    print(f"Loading manga page from {manga_path.name}...")

    images = [read_image_as_np_array(manga_path)]

    print("Loading MAGI model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        "ragavsachdeva/magi",
        trust_remote_code=True
    ).to(device)
    print("Model loaded!")

    print("Processing manga page...")
    with torch.no_grad():
        results = model.predict_detections_and_associations(images)
        text_bboxes = [x["texts"] for x in results]
        ocr_results = model.predict_ocr(images, text_bboxes)

    out_path = OUTPUT_DIR / "transcript.txt"
    model.generate_transcript_for_single_image(
        results[0],
        ocr_results[0],
        filename=str(out_path),
    )
    print(f"Transcript saved to {out_path}")


if __name__ == "__main__":
    main()
