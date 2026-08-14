#!/usr/bin/env python3
"""Convert pipeline results into a ViStoryBench-format dataset.

Reads  : <results-root>/<target>/iter<max>/bestScript.json
Writes : <dataset-root>/ViStory/<NN>/story.json
         <dataset-root>/ViStory/<NN>/image/<name>/     (empty; fill in by hand)

The conversion is done by Gemini (TEXT_MODEL) using prompt.txt. Character
reference images are never generated — only their folders are created.

Targets already present in the dataset are skipped unless --force is given, so
the script is safe to re-run and to point at new targets as they appear.

Examples
--------
    python benchmark/vistorybench/src/convert_to_vistory.py --all
    python benchmark/vistorybench/src/convert_to_vistory.py -t elf2 -t goblin
    python benchmark/vistorybench/src/convert_to_vistory.py -t elf2 --force
    python benchmark/vistorybench/src/convert_to_vistory.py --skeleton --all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:  # only the conversion path needs these; --skeleton does not
    genai = types = None

# --- Paths ---
SRC_DIR = Path(__file__).resolve().parent
BENCH_DIR = SRC_DIR.parent           # benchmark/vistorybench
BASE_DIR = SRC_DIR.parents[2]        # project root

PROMPT_FILE = SRC_DIR / "prompt.txt"

DEFAULT_RESULTS_ROOT = BASE_DIR / "results_test"
DEFAULT_DATASET_ROOT = BENCH_DIR / "data" / "dataset"
DEFAULT_OUTPUTS_ROOT = BENCH_DIR / "data" / "outputs"

DATASET_NAME = "ViStory"           # hardcoded in bench_run.py, do not rename
INDEX_FILE_NAME = "story_index.json"

VALID_TAGS = {"realistic_human", "unrealistic_human", "non_human"}
SHOT_FIELDS = (
    "Plot Correspondence",
    "Setting Description",
    "Shot Perspective Design",
    "Characters Appearing",
    "Static Shot Description",
)
CHAR_REF_RE = re.compile(r"Character\s*\[\s*(\d+)\s*\]")
UNSAFE_NAME_RE = re.compile(r"[^\w \-]", re.UNICODE)

load_dotenv(BASE_DIR / ".env")

TEXT_MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")


# --- Logging ---
def info(msg: str) -> None:
    print(f"\033[94m{msg}\033[0m")


def step(msg: str) -> None:
    print(f"\033[1;96m>> {msg}\033[0m")


def success(msg: str) -> None:
    print(f"\033[92m{msg}\033[0m")


def warn(msg: str) -> None:
    print(f"\033[93mWARN: {msg}\033[0m")


def error(msg: str) -> None:
    print(f"\033[91mERROR: {msg}\033[0m", file=sys.stderr)


def with_retry(fn, max_retries: int = 5, delay: float = 5.0, label: str | None = None):
    """Call fn(), retrying on any exception. Returns None if every attempt fails."""
    for attempt in range(1, max_retries + 1):
        try:
            if label:
                info(f"{label} (attempt {attempt}/{max_retries})")
            return fn()
        except Exception as e:  # noqa: BLE001 - transient API errors are expected here
            if attempt == max_retries:
                error(f"All {max_retries} attempts failed: {e}")
                return None
            warn(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)


# --- Input discovery ---
def find_best_script(target_dir: Path, iteration: int | None) -> Path:
    """Return the bestScript.json for the requested iteration, or the highest one."""
    if iteration is not None:
        path = target_dir / f"iter{iteration}" / "bestScript.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        return path

    candidates = []
    for d in target_dir.glob("iter*"):
        if not d.is_dir():
            continue
        m = re.match(r"^iter(\d+)$", d.name)
        if m and (d / "bestScript.json").exists():
            candidates.append((int(m.group(1)), d / "bestScript.json"))
    if not candidates:
        raise FileNotFoundError(f"No iter*/bestScript.json found under {target_dir}")
    return max(candidates, key=lambda x: x[0])[1]


def check_characters(best_script: dict) -> list:
    """Return the script's own character list, warning about unreferenced ids."""
    characters = [c for c in best_script.get("characters", []) if isinstance(c.get("id"), int)]
    known = {c["id"] for c in characters}

    missing = sorted(referenced_character_ids(best_script) - known)
    if missing:
        warn(f"clips reference Character{missing} but the script defines no such character")

    return characters


def referenced_character_ids(best_script: dict) -> set[int]:
    ids: set[int] = set()
    for clip in best_script.get("clips", []):
        blob = f"{clip.get('description', '')} {clip.get('environment', '')}"
        ids |= {int(m) for m in CHAR_REF_RE.findall(blob)}
    return ids


# --- Story id assignment ---
def load_index(dataset_dir: Path) -> dict:
    path = dataset_dir / INDEX_FILE_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_index(dataset_dir: Path, index: dict) -> None:
    path = dataset_dir / INDEX_FILE_NAME
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assign_story_id(dataset_dir: Path, target: str, requested: str | None) -> str:
    """Map a target name to a stable, zero-padded numeric story id."""
    index = load_index(dataset_dir)
    if requested:
        story_id = requested.zfill(2)
        owner = next((t for t, sid in index.items() if sid == story_id and t != target), None)
        if owner:
            raise ValueError(f"story id {story_id} is already used by target '{owner}'")
    elif target in index:
        return index[target]
    else:
        used = set(index.values()) | {d.name for d in dataset_dir.iterdir() if d.is_dir()}
        n = 1
        while f"{n:02d}" in used:
            n += 1
        story_id = f"{n:02d}"

    index[target] = story_id
    save_index(dataset_dir, index)
    return story_id


# --- Gemini calls ---
def make_client() -> "genai.Client":
    if genai is None:
        raise RuntimeError(
            "google-genai is not installed in this python. Use an env that has it "
            "(the pipeline env), or pass --skeleton to skip all model calls."
        )
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY is not set in {BASE_DIR / '.env'}")
    return genai.Client(api_key=api_key)


def build_config(**kwargs) -> types.GenerateContentConfig:
    """Build a config, degrading gracefully if the SDK lacks thinking_config."""
    try:
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"), **kwargs
        )
    except Exception:  # noqa: BLE001 - older SDK / model without thinking support
        return types.GenerateContentConfig(**kwargs)


def strip_code_block(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def build_conversion_prompt(target: str, best_script: dict) -> str:
    # The template contains literal braces, so substitute by replace(), not format().
    template = PROMPT_FILE.read_text(encoding="utf-8")
    template = template.replace("{TARGET_NAME}", target)
    template = template.replace(
        "{BEST_SCRIPT_JSON}", json.dumps(best_script, ensure_ascii=False, indent=2)
    )
    return template


def convert_script(client: genai.Client, target: str, best_script: dict) -> dict | None:
    prompt = build_conversion_prompt(target, best_script)
    config = build_config(response_mime_type="application/json")

    raw = with_retry(
        lambda: client.models.generate_content(
            model=TEXT_MODEL, contents=[prompt], config=config
        ).text,
        label=f"Converting '{target}' to ViStory schema ({TEXT_MODEL})",
    )
    if not raw:
        return None
    try:
        return json.loads(strip_code_block(raw))
    except json.JSONDecodeError as e:
        error(f"Model returned invalid JSON for '{target}': {e}")
        return None


# --- Validation / repair ---
def safe_name(name: str) -> str:
    return UNSAFE_NAME_RE.sub("", name).strip() or "Unnamed"


def validate_and_repair(story: dict, best_script: dict) -> dict:
    """Coerce the model output into something bench_run.py can actually read.

    ViStoryBench crashes or silently misscores on malformed input, so every
    constraint the evaluators rely on is enforced here rather than trusted.
    """
    if not isinstance(story, dict):
        raise ValueError("converted story is not a JSON object")

    characters = story.get("Characters")
    if not isinstance(characters, dict) or not characters:
        raise ValueError("converted story has no Characters")

    # Characters: sanitise names, key == name_en, valid tag, keep id -> name map.
    id_to_name: dict[int, str] = {}
    fixed_chars: dict[str, dict] = {}
    for raw_name, char in characters.items():
        if not isinstance(char, dict):
            continue
        name = safe_name(char.get("name_en") or raw_name)
        source_id = char.pop("source_id", None)
        if isinstance(source_id, int):
            id_to_name[source_id] = name
        char["name_en"] = name
        char.setdefault("name_ch", name)
        char.setdefault("prompt_en", "")
        char.setdefault("prompt_ch", char["prompt_en"])
        if char.get("tag") not in VALID_TAGS:
            warn(f"character '{name}' has invalid tag {char.get('tag')!r}; using 'unrealistic_human'")
            char["tag"] = "unrealistic_human"
        fixed_chars[name] = char

    # Fall back to input order when the model omitted source_id.
    if not id_to_name:
        names = list(fixed_chars)
        for c, name in zip(best_script.get("characters", []), names):
            if isinstance(c.get("id"), int):
                id_to_name[c["id"]] = name

    name_to_ch = {n: c.get("name_ch", n) for n, c in fixed_chars.items()}

    # Shots: index must equal the source clip id, so images line up by number.
    clips = {c["id"]: c for c in best_script.get("clips", []) if isinstance(c.get("id"), int)}
    shots = story.get("Shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("converted story has no Shots")

    fixed_shots = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        try:
            idx = int(shot.get("index"))
        except (TypeError, ValueError):
            warn(f"dropping shot with non-numeric index: {shot.get('index')!r}")
            continue
        shot["index"] = idx

        for field in SHOT_FIELDS:
            value = shot.get(field)
            if not isinstance(value, dict):
                value = {"en": "", "ch": ""}
            if field == "Characters Appearing":
                en = [safe_name(n) for n in value.get("en", []) if isinstance(n, str)]
                en = [n for n in en if n in fixed_chars]
                if not en:
                    # Recover from the raw clip text: Character[N] -> name.
                    clip = clips.get(idx, {})
                    blob = f"{clip.get('description', '')} {clip.get('environment', '')}"
                    ids = sorted({int(m) for m in CHAR_REF_RE.findall(blob)})
                    en = [id_to_name[i] for i in ids if i in id_to_name]
                    if en:
                        info(f"  shot {idx}: recovered Characters Appearing from clip text -> {en}")
                value = {"en": en, "ch": [name_to_ch[n] for n in en]}
            else:
                value.setdefault("en", "")
                value.setdefault("ch", value["en"])
            shot[field] = value

        fixed_shots.append(shot)

    fixed_shots.sort(key=lambda s: s["index"])

    if clips:
        expected = sorted(clips)
        got = [s["index"] for s in fixed_shots]
        if got != expected:
            warn(f"shot indices {got} do not match clip ids {expected}")
    if fixed_shots and fixed_shots[0]["index"] == 0:
        warn("shot indices start at 0; ViStoryBench expects 1-based indices")

    # num_of_appearances is recomputed rather than trusted.
    for name, char in fixed_chars.items():
        char["num_of_appearances"] = sum(
            1 for s in fixed_shots if name in s["Characters Appearing"]["en"]
        )

    story_type = story.get("Story_type")
    if not isinstance(story_type, dict):
        story_type = {"en": "", "ch": ""}
    story_type.setdefault("en", "")
    story_type.setdefault("ch", story_type["en"])

    return {"Story_type": story_type, "Characters": fixed_chars, "Shots": fixed_shots}


# --- Conversion ---
def convert_target(client: genai.Client, target: str, args) -> bool:
    step(f"Target: {target}")

    target_dir = Path(args.results_root) / target
    if not target_dir.is_dir():
        error(f"{target_dir} does not exist")
        return False

    dataset_dir = Path(args.dataset_root) / DATASET_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)

    story_id = assign_story_id(dataset_dir, target, args.story_id)
    story_dir = dataset_dir / story_id
    story_path = story_dir / "story.json"
    image_dir = story_dir / "image"
    info(f"  {target} -> story {story_id} ({story_dir})")

    if story_path.exists() and not args.force:
        info(f"  {story_path} already exists, skipping conversion. Use --force to regenerate.")
        story = json.loads(story_path.read_text(encoding="utf-8-sig"))
    else:
        best_script_path = find_best_script(target_dir, args.iter)
        info(f"  Source script: {best_script_path}")
        best_script = json.loads(best_script_path.read_text(encoding="utf-8"))
        if not check_characters(best_script):
            error(f"  '{target}' has no characters in bestScript.json")
            return False

        story = convert_script(client, target, best_script)
        if story is None:
            return False
        try:
            story = validate_and_repair(story, best_script)
        except ValueError as e:
            error(f"  Conversion for '{target}' is unusable: {e}")
            return False

        story_dir.mkdir(parents=True, exist_ok=True)
        story_path.write_text(
            json.dumps(story, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        success(f"  Saved {story_path} ({len(story['Shots'])} shots, {len(story['Characters'])} characters)")

    # One folder per character name, for reference images to be added by hand.
    image_dir.mkdir(parents=True, exist_ok=True)
    for char_name in story.get("Characters", {}):
        (image_dir / char_name).mkdir(parents=True, exist_ok=True)
        info(f"    reference images -> {image_dir / char_name}/  (00.png, 01.png, ...)")
    return True


def make_skeleton(target: str, args) -> bool:
    """Assign a story id and create the directories, without calling any model."""
    target_dir = Path(args.results_root) / target
    if not target_dir.is_dir():
        error(f"{target_dir} does not exist")
        return False

    dataset_dir = Path(args.dataset_root) / DATASET_NAME
    dataset_dir.mkdir(parents=True, exist_ok=True)
    story_id = assign_story_id(dataset_dir, target, args.story_id)
    story_dir = dataset_dir / story_id
    (story_dir / "image").mkdir(parents=True, exist_ok=True)

    have_story = (story_dir / "story.json").exists()
    info(f"  {target:<10} -> story {story_id}  "
         f"[{'story.json present' if have_story else 'story.json MISSING'}]")

    # Outputs side: one shots/ dir per mode, since the same story can be
    # evaluated from cropped storyboard panels and from video frames.
    for mode in args.mode_name:
        shots_dir = (Path(args.outputs_root) / args.method / mode / args.language
                     / args.timestamp / story_id / "shots")
        shots_dir.mkdir(parents=True, exist_ok=True)
        info(f"               shots -> {shots_dir}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert pipeline results into a ViStoryBench-format dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-t", "--target", action="append", default=[],
                        help="Target name under <results-root>/. Repeatable.")
    parser.add_argument("--all", action="store_true",
                        help="Convert every target under <results-root>/.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT),
                        help="Directory holding <target>/iter*/bestScript.json")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT),
                        help=f"Dataset root; stories are written under <dataset-root>/{DATASET_NAME}/")
    parser.add_argument("--iter", type=int, default=None,
                        help="Iteration to convert. Default: the highest iter* with a bestScript.json")
    parser.add_argument("--story-id", default=None,
                        help="Force a specific numeric story id (single target only)")
    parser.add_argument("--skeleton", action="store_true",
                        help="Only assign story ids and create directories. Makes no API calls.")
    parser.add_argument("--outputs-root", default=str(DEFAULT_OUTPUTS_ROOT),
                        help="Root for generated shot images (skeleton mode)")
    parser.add_argument("--method", default="ours",
                        help="Method name under <outputs-root>/ (skeleton mode)")
    parser.add_argument("--mode-name", action="append", default=None,
                        help="Output mode(s). Repeatable. Default: storyboard, video")
    parser.add_argument("--language", default="en", choices=["en", "ch"],
                        help="Language dimension of the outputs tree")
    parser.add_argument("--timestamp", default=None,
                        help="Timestamp dir name. Default: now, as YYYYmmdd_HHMMSS")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate story.json even if it already exists")
    args = parser.parse_args()
    if not args.mode_name:
        args.mode_name = ["storyboard", "video"]
    if not args.timestamp:
        args.timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_root = Path(args.results_root)
    if not results_root.is_dir():
        error(f"{results_root} does not exist")
        return 1

    targets = list(args.target)
    if args.all:
        targets = sorted(d.name for d in results_root.iterdir() if d.is_dir())
    if not targets:
        error("No targets given. Use -t <target> or --all.")
        return 1
    if args.story_id and len(targets) > 1:
        error("--story-id can only be used with a single target.")
        return 1

    client = None
    if args.skeleton:
        info("--skeleton: creating directories only, no model calls.")
    else:
        try:
            client = make_client()
        except RuntimeError as e:
            error(str(e))
            return 1
        info(f"Text model : {TEXT_MODEL}")

    ok, failed = [], []
    for target in targets:
        try:
            done = make_skeleton(target, args) if args.skeleton else convert_target(client, target, args)
            (ok if done else failed).append(target)
        except Exception as e:  # noqa: BLE001 - one bad target must not kill the batch
            error(f"'{target}' failed: {e}")
            failed.append(target)

    print()
    success(f"Converted: {len(ok)}/{len(targets)} -> {Path(args.dataset_root) / DATASET_NAME}")
    if failed:
        warn(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
