import json
import mimetypes
import os
import re
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.log import error, info, step, subsection, success, warn
from src.utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
FEW_SHOT_DIR = BASE_DIR / "few-shot_example"
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
PANELS_DIR = TARGET_DIR / "manga_panels"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"
CHARACTER_DIR = RESULTS_DIR / "character"
PROMPTS_DIR = BASE_DIR / "prompts"

MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")
NUM_SCRIPTS = int(os.environ.get("NUM_SCRIPTS", "3"))
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "6"))
EXP1 = int(os.environ.get("EXP1", "0"))
EXP2 = int(os.environ.get("EXP2", "3"))

PROMPT_FILES = {
    "without": PROMPTS_DIR / "create_script_without_feedback.txt",
    "one": PROMPTS_DIR / "create_script_with_one_feedback.txt",
    "two": PROMPTS_DIR / "create_script_with_two_feedback.txt",
    "three": PROMPTS_DIR / "create_script_with_three_feedback.txt",
}

REVIEWER_LABELS = ["Reviewer A", "Reviewer B", "Reviewer C"]

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


def build_prompt_without(template: str) -> str:
    for n, exp_id in enumerate([EXP1, EXP2], 1):
        char_desc = (FEW_SHOT_DIR / f"character_description{exp_id}.json").read_text(encoding="utf-8")
        transcript = (FEW_SHOT_DIR / f"transcript{exp_id}.txt").read_text(encoding="utf-8")
        script = (FEW_SHOT_DIR / f"script{exp_id}.json").read_text(encoding="utf-8")
        template = template.replace(f"{{Character Description {n}}}", char_desc)
        template = template.replace(f"{{Transcript {n}}}", transcript)
        template = template.replace(f"{{Output {n}}}", script)

    target_char = (CHARACTER_DIR / "character_description.json").read_text(encoding="utf-8")
    target_transcript = (CHARACTER_DIR / "transcript.txt").read_text(encoding="utf-8")
    template = template.replace("{Target Character Description}", target_char)
    template = template.replace("{Target Transcript}", target_transcript)
    return template


def build_prompt_with_feedback(template: str, mode: str, iteration: int) -> str:
    prev_dir = iter_dir(iteration - 1)
    best_script = (prev_dir / "bestScript.json").read_text(encoding="utf-8")
    script_feedback = (prev_dir / "intermediate" / "feedback.json").read_text(encoding="utf-8")
    sb_feedback_data = json.loads(
        (prev_dir / "intermediate" / "storyboard_feedback.json").read_text(encoding="utf-8")
    )
    sb_feedbacks = sb_feedback_data.get("feedbacks", [])

    template = template.replace("{Previously Generated Script}", best_script)
    template = template.replace("{Script Reviewer Feedback}", script_feedback)

    if mode == "one":
        if len(sb_feedbacks) < 1:
            raise ValueError(f"mode=one but iter{iteration-1} has {len(sb_feedbacks)} sb feedbacks")
        template = template.replace(
            "{Storyboard Reviewer Feedback}",
            json.dumps(sb_feedbacks[0], ensure_ascii=False, indent=2),
        )
    else:
        expected = {"two": 2, "three": 3}[mode]
        if len(sb_feedbacks) < expected:
            raise ValueError(
                f"mode={mode} requires {expected} sb feedbacks, "
                f"iter{iteration-1} has {len(sb_feedbacks)}"
            )
        for i in range(expected):
            label = REVIEWER_LABELS[i]
            template = template.replace(
                f"{{Storyboard Reviewer Feedback ({label})}}",
                json.dumps(sb_feedbacks[i], ensure_ascii=False, indent=2),
            )
    return template


def build_prompt(mode: str, iteration: int) -> str:
    template = PROMPT_FILES[mode].read_text(encoding="utf-8")
    if mode == "without":
        return build_prompt_without(template)
    return build_prompt_with_feedback(template, mode, iteration)


def generate_script(mode: str, iteration: int) -> str:
    prompt_text = build_prompt(mode, iteration)

    image_paths = []
    if mode == "without":
        image_paths.append(find_image(FEW_SHOT_DIR, f"manga_page{EXP1}"))
        image_paths.append(find_image(FEW_SHOT_DIR, f"manga_page{EXP2}"))
    for clip_id in range(1, NUM_CLIPS + 1):
        image_paths.append(PANELS_DIR / f"panel{clip_id}.png")

    parts = []
    for img_path in image_paths:
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime_type))
    parts.append(types.Part.from_text(text=prompt_text))

    return with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            ),
        ).text,
        label=f"Generating script (mode={mode}, iter={iteration})",
    )


def main(iteration: int = 0, mode: str = "without"):
    if mode not in PROMPT_FILES:
        raise ValueError(f"Unknown mode '{mode}'. Use one of {list(PROMPT_FILES)}")
    if iteration == 0 and mode != "without":
        raise ValueError("iter 0 must use mode='without'")
    if iteration > 0 and mode == "without":
        raise ValueError("iter > 0 cannot use mode='without'")

    out_dir = iter_dir(iteration) / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    subsection(f"create_script (iter={iteration}, mode={mode}, n={NUM_SCRIPTS})")

    for i in range(1, NUM_SCRIPTS + 1):
        step(f"Script {i}/{NUM_SCRIPTS}")
        raw = generate_script(mode, iteration)
        if raw is None:
            warn(f"Skipping script {i} (all retries failed)")
            continue

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            warn(f"Could not extract JSON; saving raw output")
            (out_dir / f"script_{i}_raw.txt").write_text(raw, encoding="utf-8")
            continue

        script = json.loads(json_match.group())
        out_path = out_dir / f"script_{i}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        success(f"Saved {out_path.name}")


if __name__ == "__main__":
    main()
