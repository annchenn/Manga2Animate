import json
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types

# --- Configuration ---
BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"
PANELS_DIR = BASE_DIR / "manga_panels"
PROMPT_FILE = BASE_DIR / "script_reviewer_prompt.txt"
OUTPUT_DIR = BASE_DIR / "results"

NUM_SCRIPTS = 5
NUM_CLIPS = 6
MODEL = "gemini-3.1-pro"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def load_script(script_id: int) -> dict:
    path = SCRIPTS_DIR / f"script{script_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_clip_text(script_data: dict, clip_id: int) -> str:
    clip = script_data["clips"][clip_id - 1]
    return f"Environment: {clip['environment']}\nDescription: {clip['description']}"


def evaluate_clip(clip_id: int, all_scripts: list[dict]) -> str:
    prompt_template = load_prompt_template()
    panel_path = PANELS_DIR / f"{clip_id}.png"

    # Build the script sections to fill into the prompt
    script_sections = []
    for i in range(1, NUM_SCRIPTS + 1):
        script_sections.append(build_clip_text(all_scripts[i - 1], clip_id))

    # Replace the placeholder script sections in the prompt
    filled_prompt = prompt_template
    for i in range(1, NUM_SCRIPTS + 1):
        filled_prompt = filled_prompt.replace(
            f"Script {i}:\n",
            f"Script {i}:\n{script_sections[i - 1]}\n",
            1,
        )

    print(f"  Sending request to {MODEL}...")
    image_bytes = panel_path.read_bytes()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    response = client.models.generate_content(
        model=MODEL,
        contents=[image_part, filled_prompt],
    )

    return response.text


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load all scripts
    print("Loading scripts...")
    all_scripts = []
    for i in range(1, NUM_SCRIPTS + 1):
        path = SCRIPTS_DIR / f"script{i}.json"
        if not path.exists():
            print(f"  Error: {path} not found.")
            sys.exit(1)
        all_scripts.append(load_script(i))
        print(f"  Loaded script{i}.json")

    # Evaluate clip by clip
    for clip_id in range(1, NUM_CLIPS + 1):
        print(f"\n[Clip {clip_id}/{NUM_CLIPS}] Evaluating all scripts against panel {clip_id}...")
        result = evaluate_clip(clip_id, all_scripts)

        out_path = OUTPUT_DIR / f"clip_{clip_id}_result.txt"
        out_path.write_text(result, encoding="utf-8")
        print(f"  Result saved to {out_path}")
        print(f"  Preview: {result[:200]}...")

    print("\nDone! All results saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
