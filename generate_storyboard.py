import json
import mimetypes
import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
FEW_SHOT_DIR = BASE_DIR / "few-shot_example"
TARGET_DIR = BASE_DIR / "target"
SCRIPTS_OUTPUT_DIR = BASE_DIR / "results" / "scripts"
STORYBOARD_OUTPUT_DIR = BASE_DIR / "results" / "storyboard"
STORYBOARD_PROMPT_FILE = BASE_DIR / "prompts" / "create_storyboard_prompt.txt"

MODEL = "gemini-3-pro-image-preview"
NUM_STORYBOARDS = 3

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def build_storyboard_prompt(best_script: dict) -> str:
    template = STORYBOARD_PROMPT_FILE.read_text(encoding="utf-8")
    script2 = json.loads((FEW_SHOT_DIR / "script2.json").read_text(encoding="utf-8"))
    script3 = json.loads((FEW_SHOT_DIR / "script3.json").read_text(encoding="utf-8"))

    filled = template
    filled = filled.replace(
        "Example 1 Animation Script:\n\n",
        f"Example 1 Animation Script:\n{json.dumps(script2, ensure_ascii=False, indent=2)}\n\n",
        1,
    )
    filled = filled.replace(
        "Example 2 Animation Script:\n\n",
        f"Example 2 Animation Script:\n{json.dumps(script3, ensure_ascii=False, indent=2)}\n\n",
        1,
    )
    filled = filled.replace(
        "Target Animation Script:\n\n",
        f"Target Animation Script:\n{json.dumps(best_script, ensure_ascii=False, indent=2)}\n\n",
        1,
    )
    return filled


def generate_storyboard(best_script: dict, storyboard_index: int):
    print(f"  Generating storyboard {storyboard_index + 1}/{NUM_STORYBOARDS}...")
    prompt_text = build_storyboard_prompt(best_script)

    image_paths = [
        TARGET_DIR / "manga_page.jpg",
        TARGET_DIR / "ref.png",
        FEW_SHOT_DIR / "manga_page2.png",
        FEW_SHOT_DIR / "ref2.jpg",
        FEW_SHOT_DIR / "example2.png",
        FEW_SHOT_DIR / "manga_page3.png",
        FEW_SHOT_DIR / "ref3.jpg",
        FEW_SHOT_DIR / "example3.png",
    ]

    parts = []
    for img_path in image_paths:
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime_type))
    parts.append(types.Part.from_text(text=prompt_text))

    contents = [types.Content(role="user", parts=parts)]

    config = types.GenerateContentConfig(
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="1K",
        ),
        response_modalities=["IMAGE"],
    )

    for chunk in client.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=config,
    ):
        if chunk.parts is None:
            continue
        part = chunk.parts[0]
        if part.inline_data and part.inline_data.data:
            ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
            out_path = STORYBOARD_OUTPUT_DIR / f"storyboard_{storyboard_index}{ext}"
            out_path.write_bytes(part.inline_data.data)
            print(f"    Saved to {out_path}")
        elif chunk.text:
            print(chunk.text)


def main():
    STORYBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_script_path = SCRIPTS_OUTPUT_DIR / "bestScript.json"
    if not best_script_path.exists():
        print(f"Error: {best_script_path} not found. Run evaluate_scripts.py first.")
        return
    best_script = json.loads(best_script_path.read_text(encoding="utf-8"))

    for i in range(NUM_STORYBOARDS):
        generate_storyboard(best_script, i)


if __name__ == "__main__":
    main()
