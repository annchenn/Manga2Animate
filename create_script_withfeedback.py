import json
import mimetypes
import os
import re
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
PANELS_DIR = BASE_DIR / "target" / "manga_panels"
REVIEW_DIR = BASE_DIR / "results" / "scripts_review"
OUTPUT_DIR = BASE_DIR / "results" / "scripts"
PROMPT_FILE = BASE_DIR / "prompts" / "create_script_withfeedback.txt"

MODEL = "gemini-3.1-pro-preview"
NUM_SCRIPTS = 3
NUM_CLIPS = 6

# client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)


def build_prompt(idx: int) -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")

    best_script = (REVIEW_DIR / f"bestScript{idx}.json").read_text(encoding="utf-8")
    feedback = (REVIEW_DIR / f"feedback{idx}.json").read_text(encoding="utf-8")

    template = template.replace(
        "[Best Script]\n",
        f"[Best Script]\n{best_script}\n",
        1,
    )
    template = template.replace(
        "[Feedback From Script Reviewer]\n",
        f"[Feedback From Script Reviewer]\n{feedback}\n",
        1,
    )
    return template


def generate_script(idx: int) -> str:
    prompt_text = build_prompt(idx)

    parts = []
    for clip_id in range(1, NUM_CLIPS + 1):
        parts.append(types.Part.from_bytes(
            data=(PANELS_DIR / f"{clip_id}.png").read_bytes(),
            mime_type="image/png",
        ))
    parts.append(types.Part.from_text(text=prompt_text))

    print(f"  Sending request to {MODEL}...")
    return with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    ).text)


def main(idx: int = 0, iteration: int = 1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, NUM_SCRIPTS + 1):
        print(f"\nGenerating script {i}/{NUM_SCRIPTS}...")
        raw = generate_script(idx)

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
