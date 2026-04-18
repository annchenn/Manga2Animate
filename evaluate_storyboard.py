import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
TARGET_DIR = BASE_DIR / "target"
STORYBOARD_DIR = BASE_DIR / "results" / "storyboard"
OUTPUT_DIR = BASE_DIR / "results" / "storyboard"
PROMPT_FILE = BASE_DIR / "prompts" / "storyboard_reviewer.txt"

MODEL = "gemini-3.1-pro-preview"

# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def evaluate_storyboards() -> str:
    prompt_template = load_prompt_template()

    def find_image(directory: Path, stem: str) -> Path:
        matches = list(directory.glob(f"{stem}.*"))
        if not matches:
            raise FileNotFoundError(f"No image found for '{stem}' in {directory}")
        return matches[0]

    # Image 1: manga page, Image 2: character reference
    reference_paths = [
        find_image(TARGET_DIR, "manga_page"),
        find_image(TARGET_DIR, "ref"),
    ]

    # Collect all storyboard images sorted by name
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    storyboard_paths = sorted(
        p for p in STORYBOARD_DIR.glob("storyboard_*.*")
        if p.suffix.lower() in image_extensions
    )
    if not storyboard_paths:
        print("Error: No storyboard images found in", STORYBOARD_DIR)
        return ""

    print(f"  Found {len(storyboard_paths)} storyboard(s): {[p.name for p in storyboard_paths]}")

    parts = []
    for img_path in reference_paths + storyboard_paths:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime_type))
    parts.append(types.Part.from_text(text=prompt_template))

    print(f"  Sending request to {MODEL}...")
    return with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    ).text)


def main():
    print("Evaluating storyboards...")
    result = evaluate_storyboards()
    if not result:
        return

    out_path = OUTPUT_DIR / "storyboard_evaluation.txt"
    out_path.write_text(result, encoding="utf-8")
    print(f"  Result saved to {out_path}")
    print(f"\nPreview:\n{result[:300]}...")


if __name__ == "__main__":
    main()
