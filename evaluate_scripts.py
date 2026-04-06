import json
import os
import sys
from pathlib import Path
from google import genai
from google.genai import types
import re
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).parent
SCRIPTS_DIR = BASE_DIR / "scripts"
PANELS_DIR = BASE_DIR / "manga_panels"
PROMPT_FILE = BASE_DIR / "prompts" / "script_reviewer_prompt.txt"
OUTPUT_DIR = BASE_DIR / "results" / "scripts"
SCRIPT_PER_CLIP_DIR = BASE_DIR / "script_per_clip"
NUM_SCRIPTS = 5
NUM_CLIPS = 6
MODEL = "gemini-pro-latest"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))


def load_prompt_template() -> str:
    return PROMPT_FILE.read_text(encoding="utf-8")


def load_script(script_id: int) -> dict:
    path = SCRIPTS_DIR / f"script{script_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_clip_text(script_data: dict, clip_id: int) -> str:
    clip = script_data["clips"][clip_id - 1]
    return f"environment: {clip['environment']}\ndescription: {clip['description']}"


# gemini pro evaluate script
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
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",
            ),
        ),
    )

    return response.text


def save_scripts_per_clip(clip_id: int, all_scripts: list[dict]):
    clip_data = {}
    for i, script in enumerate(all_scripts, 1):
        clip_data[f"script{i}"] = script["clips"][clip_id - 1]
    out_path = SCRIPT_PER_CLIP_DIR / f"clip{clip_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clip_data, f, ensure_ascii=False, indent=2)


def parse_scores_from_result(result_text: str) -> list[int]:
    """Extract scores for each script from the Summary section. Returns list of scores indexed 0..NUM_SCRIPTS-1."""
    summary_match = re.search(r"\*{0,2}Summary:?\*{0,2}(.*)", result_text, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary_text = summary_match.group(1)
        scores = []
        for match in re.finditer(r"Script\s+\d+\s*:\s*(\d+)\s*/\s*\d+", summary_text, re.IGNORECASE):
            scores.append(int(match.group(1)))
        if scores:
            return scores
    return []


def build_best_script(all_scripts: list[dict]) -> dict:
    """Read saved result files, pick best-scoring script per clip, assemble new script JSON."""
    best_clips = []
    for clip_id in range(1, NUM_CLIPS + 1):
        result_path = OUTPUT_DIR / f"clip_{clip_id}_result.txt"
        result_text = result_path.read_text(encoding="utf-8")
        scores = parse_scores_from_result(result_text)

        if len(scores) != NUM_SCRIPTS:
            print(f"  Warning: expected {NUM_SCRIPTS} scores for clip {clip_id}, got {len(scores)}. Defaulting to script1.")
            best_idx = 0
        else:
            best_idx = scores.index(max(scores))

        best_script_num = best_idx + 1
        print(f"  Clip {clip_id}: best script is script{best_script_num} (score {scores[best_idx] if scores else '?'})")

        clip = all_scripts[best_idx]["clips"][clip_id - 1]
        best_clip = {k: v for k, v in clip.items() if k != "visual_analysis"}
        best_clips.append(best_clip)

    return {
        "characters": all_scripts[0]["characters"],
        "clips": best_clips,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    SCRIPT_PER_CLIP_DIR.mkdir(exist_ok=True)

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
        save_scripts_per_clip(clip_id, all_scripts)
        result = evaluate_clip(clip_id, all_scripts)

        out_path = OUTPUT_DIR / f"clip_{clip_id}_result.txt"
        out_path.write_text(result, encoding="utf-8")
        print(f"  Result saved to {out_path}")
        print(f"  Preview: {result[:200]}...")

    print("\nDone! All results saved to", OUTPUT_DIR)

    # Build and save best script
    print("\nBuilding best script from highest-scoring clips...")
    best_script = build_best_script(all_scripts)
    best_path = OUTPUT_DIR / "bestScript.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_script, f, ensure_ascii=False, indent=2)
    print(f"Best script saved to {best_path}")


if __name__ == "__main__":
    main()
