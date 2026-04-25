import json
import mimetypes
import os
import re
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"
STORYBOARD_DIR = RESULTS_DIR / "storyboard"
REVIEW_DIR = RESULTS_DIR / "scripts_review"
OUTPUT_DIR = RESULTS_DIR / "storyboard"
PROMPT_FILE = BASE_DIR / "prompts" / "storyboard_reviewer.txt"

MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")

# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)


def find_image(directory: Path, stem: str) -> Path:
    matches = list(directory.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"No image found for '{stem}' in {directory}")
    return matches[0]


def evaluate_storyboard_group(script_idx: int, storyboard_paths: list[Path]) -> str:
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

    # Fill in the best script for this group
    best_script_path = REVIEW_DIR / f"bestScript{script_idx}.json"
    if best_script_path.exists():
        best_script = best_script_path.read_text(encoding="utf-8")
        prompt = prompt_template.replace("Script:\n", f"Script:\n{best_script}\n", 1)
    else:
        print(f"  Warning: bestScript{script_idx}.json not found, sending empty script.")
        prompt = prompt_template

    reference_paths = [
        find_image(TARGET_DIR, "manga_page"),
        find_image(TARGET_DIR, "ref"),
    ]

    parts = []
    for img_path in reference_paths + storyboard_paths:
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime_type))
    parts.append(types.Part.from_text(text=prompt))

    print(f"  Sending request to {MODEL} (script {script_idx}, {len(storyboard_paths)} storyboards)...")
    return with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    ).text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    all_paths = sorted(
        p for p in STORYBOARD_DIR.glob("storyboard_s*.*")
        if p.suffix.lower() in image_extensions
    )
    if not all_paths:
        print("Error: No storyboard images found in", STORYBOARD_DIR)
        return

    # Group by script_idx parsed from filename storyboard_s{idx}_{n}
    groups: dict[int, list[Path]] = {}
    for p in all_paths:
        m = re.match(r"storyboard_s(\d+)_", p.name)
        if m:
            idx = int(m.group(1))
            groups.setdefault(idx, []).append(p)

    print(f"Found {len(all_paths)} storyboard(s) in {len(groups)} group(s).")

    all_results = []
    for script_idx in sorted(groups):
        paths = groups[script_idx]
        print(f"\n[Script {script_idx}] Evaluating {len(paths)} storyboard(s): {[p.name for p in paths]}")
        result = evaluate_storyboard_group(script_idx, paths)
        if result:
            all_results.append(f"=== Script {script_idx} ===\n\n{result}")
        else:
            print(f"  Script {script_idx} evaluation failed.")

    if not all_results:
        return

    combined = "\n\n".join(all_results)
    out_path = OUTPUT_DIR / "storyboard_evaluation.txt"
    out_path.write_text(combined, encoding="utf-8")
    print(f"\nResult saved to {out_path}")
    print(f"\nPreview:\n{combined[:300]}...")


if __name__ == "__main__":
    main()
