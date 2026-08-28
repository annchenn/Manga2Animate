import os
from pathlib import Path
from google.genai import types
from dotenv import load_dotenv
from src.log import error, info, success
from src.runs import results_dir_from_env
from src.utils import find_character_ref_images, make_genai_client, with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
_target = os.environ.get("TARGET", "")
_target_root = os.environ.get("TARGET_ROOT", "target")
TARGET_DIR = BASE_DIR / _target_root / _target if _target else BASE_DIR / _target_root
RESULTS_DIR = results_dir_from_env(BASE_DIR)  # <results-root>/<target>/<run timestamp>
OUTPUT_DIR = RESULTS_DIR / "character"
PROMPT_FILE = BASE_DIR / "prompts" / "create_character_description.txt"
MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")

client = make_genai_client()


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
    ref_pairs = find_character_ref_images(TARGET_DIR)  # [(id, path), ...] sorted by id
    info(f"Reference image(s): {', '.join(p.name for _, p in ref_pairs)}")
    image_parts = []
    for _, ref_path in ref_pairs:
        mime_type = "image/jpeg" if ref_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        image_parts.append(types.Part.from_bytes(data=ref_path.read_bytes(), mime_type=mime_type))

    result = with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=[*image_parts, prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            ),
        ).text,
        label=f"Generating character description ({MODEL})",
    )

    if result is None:
        error("Failed to generate character description.")
        return

    out_path = OUTPUT_DIR / "character_description.json"
    out_path.write_text(strip_code_block(result), encoding="utf-8")
    success(f"Saved {out_path}")


if __name__ == "__main__":
    main()
