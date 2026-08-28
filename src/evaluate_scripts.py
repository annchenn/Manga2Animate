import json
import os
import re
import sys
from pathlib import Path
from google.genai import types
from dotenv import load_dotenv
from src.log import error, info, step, subsection, success, warn
from src.runs import results_dir_from_env
from src.utils import find_character_ref_images, make_genai_client, with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
_target = os.environ.get("TARGET", "")
_target_root = os.environ.get("TARGET_ROOT", "target")
TARGET_DIR = BASE_DIR / _target_root / _target if _target else BASE_DIR / _target_root
PANELS_DIR = TARGET_DIR / "manga_panels"
RESULTS_DIR = results_dir_from_env(BASE_DIR)  # <results-root>/<target>/<run timestamp>

CHARACTER_PROMPT_FILE = BASE_DIR / "prompts" / "character_reviewer.txt"
CLIP_PROMPT_FILE = BASE_DIR / "prompts" / "clip_reviewer.txt"

NUM_SCRIPTS = int(os.environ.get("NUM_SCRIPTS", "3"))
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "6"))
MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")

client = make_genai_client()


def iter_dir(iteration: int) -> Path:
    return RESULTS_DIR / f"iter{iteration}"


def load_script(iteration: int, script_id: int) -> dict:
    path = iter_dir(iteration) / "scripts" / f"script_{script_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_code_block(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def parse_script_index(script_label: str) -> int:
    """ "Script 2" -> 1 (zero-based index) """
    m = re.search(r"\d+", script_label or "")
    return (int(m.group()) - 1) if m else 0


# --- Character evaluation ---

def evaluate_character(all_scripts: list[dict]) -> str:
    template = CHARACTER_PROMPT_FILE.read_text(encoding="utf-8")

    characters_payload = {
        f"Script {i + 1}": all_scripts[i].get("characters", [])
        for i in range(len(all_scripts))
    }
    filled_prompt = template.replace(
        "{json}",
        json.dumps(characters_payload, ensure_ascii=False, indent=2),
    )

    panel_parts = []
    for clip_id in range(1, NUM_CLIPS + 1):
        panel_path = PANELS_DIR / f"panel{clip_id}.png"
        panel_parts.append(types.Part.from_bytes(data=panel_path.read_bytes(), mime_type="image/png"))

    ref_parts = []
    for _, ref_path in find_character_ref_images(TARGET_DIR):
        ref_mime = "image/jpeg" if ref_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        ref_parts.append(types.Part.from_bytes(data=ref_path.read_bytes(), mime_type=ref_mime))

    return with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=[*panel_parts, *ref_parts, filled_prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            ),
        ).text,
        label=f"Character reviewer ({MODEL})",
    )


# --- Clip evaluation ---

def evaluate_clip(clip_id: int, all_scripts: list[dict]) -> str:
    template = CLIP_PROMPT_FILE.read_text(encoding="utf-8")
    panel_path = PANELS_DIR / f"panel{clip_id}.png"

    payload = {
        "Character Description": all_scripts[0].get("characters", []),
    }
    for i in range(len(all_scripts)):
        clips = all_scripts[i].get("clips", [])
        if clip_id - 1 < len(clips):
            clip = clips[clip_id - 1]
            entry = {
                "environment": clip.get("environment", ""),
                "description": clip.get("description", ""),
            }
            # flux2-backend scripts keep dialogue in its own field (see
            # create_script_with_one_feedback_flux2.txt / _no_fewshot.txt)
            # instead of inline inside "description" — pass it through so the
            # "Dialogue Speaker Assignment" criterion in clip_reviewer.txt still
            # has something to compare. gemini-backend clips never have this
            # key, so this is a no-op for them (identical payload to before).
            if "dialogue" in clip:
                entry["dialogue"] = clip["dialogue"]
            payload[f"Script {i + 1}"] = entry
        else:
            payload[f"Script {i + 1}"] = "(clip not present in this script)"

    filled_prompt = template.replace(
        "{Script}",
        json.dumps(payload, ensure_ascii=False, indent=2),
    )

    image_part = types.Part.from_bytes(data=panel_path.read_bytes(), mime_type="image/png")

    return with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=[image_part, filled_prompt],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            ),
        ).text,
        label=f"Clip {clip_id} reviewer ({MODEL})",
    )


# --- Best-script aggregation ---

def best_script_idx_from_character(char_result: dict) -> int:
    winner_label = char_result.get("overall_winner", {}).get("script")
    if winner_label:
        return parse_script_index(winner_label)
    scores = char_result.get("scores", {})
    if scores:
        return parse_script_index(max(scores, key=lambda k: scores.get(k, 0)))
    return 0


def best_script_idx_from_clip(clip_result: dict, valid_indices: list[int]) -> int:
    winner_label = clip_result.get("Highest Script")
    if winner_label:
        idx = parse_script_index(winner_label)
        if idx in valid_indices:
            return idx
    scoreboard = clip_result.get("Scoreboard", {})
    if scoreboard:
        candidates = [(parse_script_index(k), v) for k, v in scoreboard.items()]
        candidates = [(i, _to_int(v)) for i, v in candidates if i in valid_indices]
        if candidates:
            return max(candidates, key=lambda t: t[1])[0]
    return valid_indices[0]


def _to_int(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"\d+", value)
        return int(m.group()) if m else 0
    return 0


def build_best_script(iteration: int, all_scripts: list[dict]) -> dict:
    interim = iter_dir(iteration) / "intermediate"
    char_result = json.loads((interim / "character.json").read_text(encoding="utf-8"))
    best_char_idx = best_script_idx_from_character(char_result)
    info(f"Characters → Script {best_char_idx + 1}")
    best_characters = all_scripts[best_char_idx].get("characters", [])

    best_clips = []
    for clip_id in range(1, NUM_CLIPS + 1):
        result = json.loads((interim / f"clip_{clip_id}.json").read_text(encoding="utf-8"))
        valid_indices = [
            i for i in range(len(all_scripts))
            if clip_id - 1 < len(all_scripts[i].get("clips", []))
        ]
        if not valid_indices:
            warn(f"Clip {clip_id}: no script has this clip, skipping.")
            continue
        best_idx = best_script_idx_from_clip(result, valid_indices)
        info(f"Clip {clip_id} → Script {best_idx + 1}")
        clip = all_scripts[best_idx]["clips"][clip_id - 1]
        best_clip = {k: v for k, v in clip.items() if k != "visual_analysis"}
        best_clips.append(best_clip)

    return {"characters": best_characters, "clips": best_clips}


def build_feedback(iteration: int, all_scripts: list[dict]) -> dict:
    interim = iter_dir(iteration) / "intermediate"
    char_result = json.loads((interim / "character.json").read_text(encoding="utf-8"))
    char_feedback = char_result.get("improvement_suggestions_for_winner", {})

    clips_feedback = []
    for clip_id in range(1, NUM_CLIPS + 1):
        result = json.loads((interim / f"clip_{clip_id}.json").read_text(encoding="utf-8"))
        clips_feedback.append({
            "clip_id": clip_id,
            "feedback": result.get("improvement_suggestions_for_winner", {}),
        })

    return {
        "character": {"feedback": char_feedback},
        "clips": clips_feedback,
    }


# --- Main ---

def main(iteration: int = 0):
    out_dir = iter_dir(iteration)
    interim = out_dir / "intermediate"
    interim.mkdir(parents=True, exist_ok=True)

    subsection(f"evaluate_scripts (iter={iteration})")

    info(f"Loading {NUM_SCRIPTS} scripts...")
    all_scripts = []
    for i in range(1, NUM_SCRIPTS + 1):
        path = out_dir / "scripts" / f"script_{i}.json"
        if not path.exists():
            error(f"{path} not found.")
            sys.exit(1)
        all_scripts.append(load_script(iteration, i))
    success(f"Loaded {len(all_scripts)} scripts")

    step("Character review")
    char_result = evaluate_character(all_scripts)
    if char_result is None:
        warn("Character review failed; defaulting to Script 1.")
        char_result = '{"overall_winner": {"script": "Script 1"}, "scores": {"Script 1": 0}}'
    (interim / "character.json").write_text(strip_code_block(char_result), encoding="utf-8")
    success(f"Saved {interim / 'character.json'}")

    for clip_id in range(1, NUM_CLIPS + 1):
        step(f"Clip {clip_id}/{NUM_CLIPS} review")
        result = evaluate_clip(clip_id, all_scripts)
        if result is None:
            warn(f"Clip {clip_id} review failed; defaulting to Script 1.")
            result = '{"Highest Script": "Script 1", "Scoreboard": {"Script 1": 0}}'
        out_path = interim / f"clip_{clip_id}.json"
        out_path.write_text(strip_code_block(result), encoding="utf-8")
        success(f"Saved {out_path.name}")

    step("Building best script")
    best_script = build_best_script(iteration, all_scripts)
    best_path = out_dir / "bestScript.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_script, f, ensure_ascii=False, indent=2)
    success(f"Saved {best_path}")

    step("Extracting feedback")
    feedback = build_feedback(iteration, all_scripts)
    feedback_path = interim / "feedback.json"
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    success(f"Saved {feedback_path}")


if __name__ == "__main__":
    main()
