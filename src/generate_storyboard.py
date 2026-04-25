import json
import mimetypes
import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.log import error, info, status, step, subsection, success, warn
from src.utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
FEW_SHOT_DIR = BASE_DIR / "few-shot_example"
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"
PROMPT_FILE = BASE_DIR / "prompts" / "create_storyboard.txt"

MODEL = os.environ.get("IMAGE_MODEL", "gemini-3-pro-image-preview")
NUM_STORYBOARDS = int(os.environ.get("NUM_STORYBOARDS", "3"))
EXP1 = int(os.environ.get("EXP1", "0"))
EXP2 = int(os.environ.get("EXP2", "3"))

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


def iter_dir(iteration: int) -> Path:
    return RESULTS_DIR / f"iter{iteration}"


def build_storyboard_prompt(best_script: dict) -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")
    script_exp1 = json.loads((FEW_SHOT_DIR / f"script{EXP1}.json").read_text(encoding="utf-8"))
    script_exp2 = json.loads((FEW_SHOT_DIR / f"script{EXP2}.json").read_text(encoding="utf-8"))

    template = template.replace(
        "{Example 1 Animation Script}",
        json.dumps(script_exp1, ensure_ascii=False, indent=2),
    )
    template = template.replace(
        "{Example 2 Animation Script}",
        json.dumps(script_exp2, ensure_ascii=False, indent=2),
    )
    template = template.replace(
        "{Target Animation Script}",
        json.dumps(best_script, ensure_ascii=False, indent=2),
    )
    return template


def generate_one(best_script: dict, iteration: int, sb_index: int):
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

    out_dir = iter_dir(iteration) / "storyboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

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
                out_path = out_dir / f"storyboard_{sb_index}{ext}"
                out_path.write_bytes(part.inline_data.data)
                saved_paths.append(out_path)

    with_retry(_stream, label=f"Generating storyboard {sb_index} ({MODEL})")
    for p in saved_paths:
        success(f"Saved {p}")


def main(iteration: int = 0):
    best_path = iter_dir(iteration) / "bestScript.json"
    if not best_path.exists():
        error(f"{best_path} not found. Run evaluate_scripts first.")
        return
    best_script = json.loads(best_path.read_text(encoding="utf-8"))

    subsection(f"generate_storyboard (iter={iteration}, n={NUM_STORYBOARDS})")

    for i in range(NUM_STORYBOARDS):
        step(f"Storyboard {i + 1}/{NUM_STORYBOARDS}")
        generate_one(best_script, iteration, i)


if __name__ == "__main__":
    main()
