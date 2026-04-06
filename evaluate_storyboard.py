import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
TARGET_DIR = BASE_DIR / "target"
STORYBOARD_DIR = BASE_DIR / "results" / "storyboard"
OUTPUT_DIR = BASE_DIR / "results" / "storyboard"
PROMPT_FILE = BASE_DIR / "prompts" / "storyboard_reviewer_prompt.txt"

MODEL = "gemini-pro-latest"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def evaluate_storyboards() -> str:
    prompt_template = load_prompt_template()

    # Image 1: manga page, Image 2: character reference
    reference_paths = [
        TARGET_DIR / "manga_page.jpg",
        TARGET_DIR / "ref.png",
    ]

    # Collect all storyboard images sorted by name
    storyboard_paths = sorted(STORYBOARD_DIR.glob("storyboard_*.*"))
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
    response = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    )
    return response.text


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
