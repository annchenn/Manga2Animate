import json
import mimetypes
import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
FEW_SHOT_DIR = BASE_DIR / "few-shot_example"
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"
SCRIPTS_OUTPUT_DIR = RESULTS_DIR / "scripts_review"
STORYBOARD_OUTPUT_DIR = RESULTS_DIR / "storyboard"
STORYBOARD_PROMPT_FILE = BASE_DIR / "prompts" / "create_storyboard.txt"

MODEL = os.environ.get("IMAGE_MODEL", "gemini-3-pro-image-preview")
NUM_STORYBOARDS = int(os.environ.get("NUM_STORYBOARDS", "3"))
EXP1 = int(os.environ.get("EXP1", "2"))
EXP2 = int(os.environ.get("EXP2", "3"))

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


def build_storyboard_prompt(best_script: dict) -> str:
    template = STORYBOARD_PROMPT_FILE.read_text(encoding="utf-8")
    script_exp1 = json.loads((FEW_SHOT_DIR / f"script{EXP1}.json").read_text(encoding="utf-8"))
    script_exp2 = json.loads((FEW_SHOT_DIR / f"script{EXP2}.json").read_text(encoding="utf-8"))

    filled = template
    filled = filled.replace(
        "Example 1 Animation Script:\n\n",
        f"Example 1 Animation Script:\n{json.dumps(script_exp1, ensure_ascii=False, indent=2)}\n\n",
        1,
    )
    filled = filled.replace(
        "Example 2 Animation Script:\n\n",
        f"Example 2 Animation Script:\n{json.dumps(script_exp2, ensure_ascii=False, indent=2)}\n\n",
        1,
    )
    filled = filled.replace(
        "Target Animation Script:\n\n",
        f"Target Animation Script:\n{json.dumps(best_script, ensure_ascii=False, indent=2)}\n\n",
        1,
    )
    return filled


def generate_storyboard(best_script: dict, storyboard_index: int, script_idx: int = 0):
    print(f"  Generating storyboard {storyboard_index + 1}/{NUM_STORYBOARDS}...")
    prompt_text = build_storyboard_prompt(best_script)

    image_paths = [
        find_image(TARGET_DIR, "manga_page"),
        find_image(TARGET_DIR, "ref"),
        find_image(FEW_SHOT_DIR, f"manga_page{EXP1}"),
        find_image(FEW_SHOT_DIR, f"ref{EXP1}"),
        find_image(FEW_SHOT_DIR, f"example{EXP1}"),
        find_image(FEW_SHOT_DIR, f"manga_page{EXP2}"),
        find_image(FEW_SHOT_DIR, f"ref{EXP2}"),
        find_image(FEW_SHOT_DIR, f"example{EXP2}"),
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

    def _stream():
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
                out_path = STORYBOARD_OUTPUT_DIR / f"storyboard_s{script_idx}_{storyboard_index}{ext}"
                out_path.write_bytes(part.inline_data.data)
                print(f"    Saved to {out_path}")
            elif chunk.text:
                print(chunk.text)

    with_retry(_stream)


def main(idx: int = 2):
    STORYBOARD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_script_path = SCRIPTS_OUTPUT_DIR / f"bestScript{idx}.json"
    if not best_script_path.exists():
        print(f"Error: {best_script_path} not found. Run evaluate_scripts.py first.")
        return
    best_script = json.loads(best_script_path.read_text(encoding="utf-8"))

    for i in range(NUM_STORYBOARDS):
        print(f"\nStoryboard {i + 1}/{NUM_STORYBOARDS}")
        generate_storyboard(best_script, i, script_idx=idx)


if __name__ == "__main__":
    main()
