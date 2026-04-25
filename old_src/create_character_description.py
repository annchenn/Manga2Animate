import os
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
OUTPUT_DIR = RESULTS_DIR / "character"
PROMPT_FILE = BASE_DIR / "prompts" / "create_character_description.txt"
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


def strip_code_block(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    ref_path = find_image(TARGET_DIR, "ref")
    mime_type = "image/jpeg" if ref_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    image_part = types.Part.from_bytes(data=ref_path.read_bytes(), mime_type=mime_type)

    print(f"Generating character description from {ref_path.name}...")
    result = with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=[image_part, prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    ).text)

    if result is None:
        print("Failed to generate character description.")
        return

    out_path = OUTPUT_DIR / "character_description.json"
    out_path.write_text(strip_code_block(result), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
