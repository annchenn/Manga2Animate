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
FEW_SHOT_DIR = BASE_DIR / "few-shot_example"
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
PANELS_DIR = TARGET_DIR / "manga_panels"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "scripts"
CHARACTER_DIR = RESULTS_DIR / "character"
PROMPT_FILE = BASE_DIR / "prompts" / "create_script_withoutfeedback.txt"

MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")
NUM_SCRIPTS = int(os.environ.get("NUM_SCRIPTS", "3"))
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "6"))
EXP1 = int(os.environ.get("EXP1", "0"))
EXP2 = int(os.environ.get("EXP2", "2"))

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


def build_prompt() -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")

    for n, exp_id in enumerate([EXP1, EXP2], 1):
        char_desc = (FEW_SHOT_DIR / f"character_description{exp_id}.json").read_text(encoding="utf-8")
        transcript = (FEW_SHOT_DIR / f"transcript{exp_id}.txt").read_text(encoding="utf-8")
        script = (FEW_SHOT_DIR / f"script{exp_id}.json").read_text(encoding="utf-8")

        template = template.replace(
            f"[Character Description {n}]\n",
            f"[Character Description {n}]\n{char_desc}\n",
            1,
        )
        template = template.replace(
            f"[Transcript {n}]\n",
            f"[Transcript {n}]\n{transcript}\n",
            1,
        )
        template = template.replace(
            f"[Output {n}]\n",
            f"[Output {n}]\n{script}\n",
            1,
        )

    target_char = (CHARACTER_DIR / "character_description.json").read_text(encoding="utf-8")
    target_transcript = (CHARACTER_DIR / "transcript.txt").read_text(encoding="utf-8")

    template = template.replace(
        "[Target Character Description]\n",
        f"[Target Character Description]\n{target_char}\n",
        1,
    )
    template = template.replace(
        "[Target Transcript]\n",
        f"[Target Transcript]\n{target_transcript}\n",
        1,
    )

    return template


def generate_script() -> str:
    prompt_text = build_prompt()

    # Image order: few-shot manga pages, then target individual panels
    image_paths = [
        find_image(FEW_SHOT_DIR, f"manga_page{EXP1}"),
        find_image(FEW_SHOT_DIR, f"manga_page{EXP2}"),
    ]
    for clip_id in range(1, NUM_CLIPS + 1):
        image_paths.append(PANELS_DIR / f"{clip_id}.png")

    parts = []
    for img_path in image_paths:
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime_type))
    parts.append(types.Part.from_text(text=prompt_text))

    print(f"  Sending request to {MODEL} (EXP1={EXP1}, EXP2={EXP2})...")
    return with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    ).text)


def main(iteration: int = 0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, NUM_SCRIPTS + 1):
        print(f"\nGenerating script {i}/{NUM_SCRIPTS}...")
        raw = generate_script()

        if raw is None:
            print(f"  Skipping script {i} due to repeated failures.")
            continue

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            print(f"  Warning: Could not extract JSON. Saving raw output.")
            (OUTPUT_DIR / f"script{iteration}_{i}_raw.txt").write_text(raw, encoding="utf-8")
            continue

        script = json.loads(json_match.group())
        out_path = OUTPUT_DIR / f"script{iteration}_{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
