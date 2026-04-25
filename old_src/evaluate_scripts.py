import json
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv
from src.utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
PANELS_DIR = TARGET_DIR / "manga_panels"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"
SCRIPTS_DIR = RESULTS_DIR / "scripts"
PROMPT_FILE = BASE_DIR / "prompts" / "clip_reviewer.txt"
CHARACTER_PROMPT_FILE = BASE_DIR / "prompts" / "character_reviewer.txt"
OUTPUT_DIR = RESULTS_DIR / "scripts_review"
NUM_SCRIPTS = int(os.environ.get("NUM_SCRIPTS", "3"))
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "6"))
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


def load_script(script_id: int, iteration: int = 0) -> dict:
    path = SCRIPTS_DIR / f"script{iteration}_{script_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_code_block(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def build_clip_text(script_data: dict, clip_id: int) -> str:
    clips = script_data.get("clips", [])
    if clip_id - 1 >= len(clips):
        return f"(clip {clip_id} not found in this script)"
    clip = clips[clip_id - 1]
    return f"environment: {clip['environment']}\ndescription: {clip['description']}"


def format_characters(script: dict) -> str:
    return "\n".join(
        f"Character {c['id']}: {c['description']}"
        for c in script.get("characters", [])
    )



# --- Clip evaluation ---

def evaluate_clip(clip_id: int, all_scripts: list[dict]) -> str:
    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    panel_path = PANELS_DIR / f"{clip_id}.png"

    filled_prompt = prompt_template
    filled_prompt = filled_prompt.replace(
        "Character:\n",
        f"Character:\n{format_characters(all_scripts[0])}\n",
        1,
    )
    for i in range(1, NUM_SCRIPTS + 1):
        clip_text = build_clip_text(all_scripts[i - 1], clip_id)
        filled_prompt = filled_prompt.replace(
            f"Script {i}:\n",
            f"Script {i}:\n{clip_text}\n",
            1,
        )

    print(f"  Sending request to {MODEL}...")
    image_part = types.Part.from_bytes(data=panel_path.read_bytes(), mime_type="image/png")

    return with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=[image_part, filled_prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    ).text)


# --- Character evaluation ---

def evaluate_character(all_scripts: list[dict]) -> str:
    prompt_template = CHARACTER_PROMPT_FILE.read_text(encoding="utf-8")

    filled_prompt = prompt_template
    for i in range(1, NUM_SCRIPTS + 1):
        char_text = format_characters(all_scripts[i - 1])
        filled_prompt = filled_prompt.replace(
            f"Script {i}:\n",
            f"Script {i}:\n{char_text}\n",
            1,
        )

    manga_path = find_image(TARGET_DIR, "manga_page")
    manga_mime = "image/jpeg" if manga_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    panel_part = types.Part.from_bytes(data=manga_path.read_bytes(), mime_type=manga_mime)

    ref_path = find_image(TARGET_DIR, "ref")
    ref_mime = "image/jpeg" if ref_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    ref_part = types.Part.from_bytes(data=ref_path.read_bytes(), mime_type=ref_mime)

    print(f"  Sending character review request to {MODEL}...")
    return with_retry(lambda: client.models.generate_content(
        model=MODEL,
        contents=[panel_part, ref_part, filled_prompt],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        ),
    ).text)


# --- Build best script ---

def get_best_idx(result: dict, valid_indices: list[int] = None) -> int:
    """Pick best script by parsing individual scores. Only consider valid_indices if provided."""
    if valid_indices is None:
        valid_indices = list(range(NUM_SCRIPTS))

    best_score = -1
    best_idx = valid_indices[0]
    for idx in valid_indices:
        score_str = result.get(f"Script {idx + 1}", {}).get("Score", "0 / 6")
        match = re.search(r"\d+", str(score_str))
        score = int(match.group()) if match else 0
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def build_best_script(all_scripts: list[dict], iteration: int = 0) -> dict:
    # Best characters from character review
    char_result = json.loads((OUTPUT_DIR / f"character{iteration}.json").read_text(encoding="utf-8"))
    best_char_idx = get_best_idx(char_result)
    print(f"  Characters: best script is Script{best_char_idx + 1}")
    best_characters = all_scripts[best_char_idx]["characters"]

    # Best clip from each clip review
    best_clips = []
    for clip_id in range(1, NUM_CLIPS + 1):
        result = json.loads((OUTPUT_DIR / f"clip_{clip_id}_{iteration}.json").read_text(encoding="utf-8"))

        # Force score to 0 for scripts that don't have this clip
        for i in range(NUM_SCRIPTS):
            if clip_id - 1 >= len(all_scripts[i]["clips"]):
                result[f"Script {i + 1}"] = {"Score": "0 / 7"}

        valid_indices = [
            i for i in range(NUM_SCRIPTS)
            if clip_id - 1 < len(all_scripts[i]["clips"])
        ]
        if not valid_indices:
            print(f"  Clip {clip_id}: no script has this clip, skipping.")
            continue

        best_idx = get_best_idx(result, valid_indices)
        print(f"  Clip {clip_id}: best script is Script{best_idx + 1}")

        clip = all_scripts[best_idx]["clips"][clip_id - 1]
        best_clip = {k: v for k, v in clip.items() if k != "visual_analysis"}
        best_clips.append(best_clip)

    return {
        "characters": best_characters,
        "clips": best_clips,
    }


# --- Build feedback ---

def build_feedback(iteration: int = 0) -> dict:
    char_result = json.loads((OUTPUT_DIR / f"character{iteration}.json").read_text(encoding="utf-8"))
    best_char_idx = get_best_idx(char_result)
    best_char_key = f"Script {best_char_idx + 1}"
    char_feedback = char_result.get(best_char_key, {})

    clips_feedback = []
    all_scripts = [load_script(i, iteration) for i in range(1, NUM_SCRIPTS + 1)]
    for clip_id in range(1, NUM_CLIPS + 1):
        result = json.loads((OUTPUT_DIR / f"clip_{clip_id}_{iteration}.json").read_text(encoding="utf-8"))
        valid_indices = [
            i for i in range(NUM_SCRIPTS)
            if clip_id - 1 < len(all_scripts[i]["clips"])
        ]
        best_idx = get_best_idx(result, valid_indices) if valid_indices else 0
        best_key = f"Script {best_idx + 1}"
        clips_feedback.append({
            "clip_id": clip_id,
            "feedback": result.get(best_key, {}),
        })

    return {
        "character": {
            "feedback": char_feedback,
        },
        "clips": clips_feedback,
    }


# --- Main ---

def main(iteration: int = 0):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading scripts...")
    all_scripts = []
    for i in range(1, NUM_SCRIPTS + 1):
        path = SCRIPTS_DIR / f"script{iteration}_{i}.json"
        if not path.exists():
            print(f"  Error: {path} not found.")
            sys.exit(1)
        all_scripts.append(load_script(i, iteration))
        print(f"  Loaded script{iteration}_{i}.json")

    # Evaluate character descriptions
    print("\n[Character] Evaluating character descriptions across all scripts...")
    char_result = evaluate_character(all_scripts)
    if char_result is None:
        print("  Character review failed. Defaulting to Script 1.")
        char_result = '{"Highest Script": "Script 1"}'
    char_path = OUTPUT_DIR / f"character{iteration}.json"
    char_path.write_text(strip_code_block(char_result), encoding="utf-8")
    print(f"  Character review saved to {char_path}")

    # Evaluate clip by clip
    for clip_id in range(1, NUM_CLIPS + 1):
        print(f"\n[Clip {clip_id}/{NUM_CLIPS}] Evaluating all scripts against panel {clip_id}...")
        result = evaluate_clip(clip_id, all_scripts)
        if result is None:
            print(f"  Clip {clip_id} review failed. Defaulting to Script 1.")
            result = '{"Highest Script": "Script 1"}'
        cleaned = strip_code_block(result)
        out_path = OUTPUT_DIR / f"clip_{clip_id}_{iteration}.json"
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"  Result saved to {out_path}")

    print("\nDone! All results saved to", OUTPUT_DIR)

    # Build best script
    print("\nBuilding best script...")
    best_script = build_best_script(all_scripts, iteration)
    best_path = OUTPUT_DIR / f"bestScript{iteration}.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_script, f, ensure_ascii=False, indent=2)
    print(f"Best script saved to {best_path}")


    # Build feedback
    print("\nExtracting feedback...")
    feedback = build_feedback(iteration)
    feedback_path = OUTPUT_DIR / f"feedback{iteration}.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print(f"Feedback saved to {feedback_path}")



if __name__ == "__main__":
    main()
