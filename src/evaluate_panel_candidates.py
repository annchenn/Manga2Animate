"""
Per-panel Gemini reviewer for the flux2 storyboard backend's seed-variants.

Each variant is scored by its OWN separate Gemini call (one prompt sees exactly
one candidate image, never multiple candidates at once — judging one candidate at
a time is more reliable than asking one prompt to compare several images). The
winner is picked by comparing the returned scores in Python, not by asking Gemini
to compare N images itself.
"""
import json
import mimetypes
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types

from src.log import info, step, subsection, success, warn
from src.runs import results_dir_from_env
from src.utils import find_character_ref_images, make_genai_client, with_retry

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
_target = os.environ.get("TARGET", "")
_target_root = os.environ.get("TARGET_ROOT", "target")
TARGET_DIR = BASE_DIR / _target_root / _target if _target else BASE_DIR / _target_root
PANELS_DIR = TARGET_DIR / "manga_panels"
RESULTS_DIR = results_dir_from_env(BASE_DIR)

PROMPT_FILE = BASE_DIR / "prompts" / "panel_candidate_reviewer.txt"
MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")

client = make_genai_client()

PRIMARY_CRITERION = "Manga/Script Follow"


def iter_dir(iteration: int) -> Path:
    return RESULTS_DIR / f"iter{iteration}"


def strip_code_block(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def part_from_image(path: Path) -> types.Part:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def build_prompt(clip: dict, char_ids_in_panel: list[int], num_refs: int) -> str:
    """Builds the prompt for scoring ONE candidate image (never multiple candidates
    in one prompt — each variant is judged independently, and Python compares the
    resulting scores afterward)."""
    template = PROMPT_FILE.read_text(encoding="utf-8")
    replacements = {
        "{Panel ID}": str(clip["id"]),
        "{Clip Environment}": clip.get("environment", ""),
        "{Clip Description}": clip.get("description", ""),
        "{Characters In Panel}": (
            ", ".join(f"Character[{i}]" for i in char_ids_in_panel) if char_ids_in_panel else "none"
        ),
        "{Num Reference Images}": str(num_refs),
    }
    for k, v in replacements.items():
        template = template.replace(k, v)
    return template


def call_reviewer(prompt: str, image_paths: list[Path], spinner_label: str) -> dict:
    parts = [part_from_image(p) for p in image_paths]
    parts.append(types.Part.from_text(text=prompt))
    raw = with_retry(
        lambda: client.models.generate_content(
            model=MODEL,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            ),
        ).text,
        label=f"{spinner_label} ({MODEL})",
    )
    if raw is None:
        return {}
    cleaned = strip_code_block(raw)
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {"_raw": cleaned}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {"_raw": cleaned}


def score_variant(
    clip: dict, candidate_path: Path, source_panel_path: Path, ref_paths: list[Path], variant_label: str
) -> dict:
    """One Gemini call, scoring exactly ONE candidate image against the 4 criteria."""
    char_ids = clip.get("characters_in_panel") or []
    prompt = build_prompt(clip, char_ids, len(ref_paths))
    images = [source_panel_path, *ref_paths, candidate_path]
    return call_reviewer(prompt, images, f"panel {clip['id']} {variant_label} scorer")


def _total_score(result: dict) -> float:
    v = result.get("total_score")
    if isinstance(v, (int, float)):
        return v
    scores = result.get("criteria_scores", {})
    vals = [
        c.get("score") for c in scores.values()
        if isinstance(c, dict) and isinstance(c.get("score"), (int, float))
    ]
    return sum(vals) if vals else -1.0


def _primary_score(result: dict) -> float:
    crit = result.get("criteria_scores", {}).get(PRIMARY_CRITERION, {})
    v = crit.get("score") if isinstance(crit, dict) else None
    return v if isinstance(v, (int, float)) else -1.0


def evaluate_panel(
    clip: dict, candidate_paths: list[Path], source_panel_path: Path, ref_paths: list[Path], out_path: Path
) -> dict:
    """Scores every candidate independently (one Gemini call each), then picks the
    winner IN PYTHON by comparing total_score (tie-break: primary-criterion score).
    Writes and returns a combined JSON: {"per_variant": {...}, "total_scores": {...},
    "overall_winner": {"variant": "variant_N"}}."""
    per_variant: dict[str, dict] = {}
    for idx, cpath in enumerate(candidate_paths, start=1):
        label = f"variant_{idx}"
        per_variant[label] = score_variant(clip, cpath, source_panel_path, ref_paths, label)

    ranked = sorted(
        per_variant.items(),
        key=lambda kv: (_total_score(kv[1]), _primary_score(kv[1])),
        reverse=True,
    )
    winner_key = ranked[0][0] if ranked else "variant_1"
    if ranked and ranked[0][1] == -1.0:
        warn(f"Panel {clip['id']}: could not parse any variant's score; defaulting to {winner_key}")

    combined = {
        "per_variant": per_variant,
        "total_scores": {k: _total_score(v) for k, v in per_variant.items()},
        "overall_winner": {"variant": winner_key},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    success(f"Saved {out_path.name}")
    return combined


def main(iteration: int = 0) -> dict[int, int]:
    """Returns {panel_id: winning_variant}. Writes candidates/panel_review_{id}.json
    (one combined file per panel, containing every variant's independent score) and
    copies each winner to storyboard/selected/panel{id}.png."""
    out_dir = iter_dir(iteration)
    best_script = json.loads((out_dir / "bestScript.json").read_text(encoding="utf-8"))
    storyboard_dir = out_dir / "storyboard"
    candidates_dir = storyboard_dir / "candidates"
    selected_dir = storyboard_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)

    ref_images = dict(find_character_ref_images(TARGET_DIR))

    subsection(f"evaluate_panel_candidates (iter={iteration})")
    winners: dict[int, int] = {}
    for clip in sorted(best_script["clips"], key=lambda c: c["id"]):
        pid = clip["id"]
        step(f"Panel {pid}")
        candidate_paths = sorted(
            candidates_dir.glob(f"panel{pid}_v*.png"),
            key=lambda p: int(re.search(r"_v(\d+)\.", p.name).group(1)),
        )
        if not candidate_paths:
            warn(f"No candidates found for panel {pid}; skipping")
            continue

        char_ids = clip.get("characters_in_panel") or []
        ref_paths = [ref_images[i] for i in char_ids if i in ref_images]

        out_json = candidates_dir / f"panel_review_{pid}.json"
        combined = evaluate_panel(clip, candidate_paths, PANELS_DIR / f"panel{pid}.png", ref_paths, out_json)
        winner_v = int(re.match(r"variant_(\d+)$", combined["overall_winner"]["variant"]).group(1))
        winners[pid] = winner_v

        winner_path = candidates_dir / f"panel{pid}_v{winner_v}.png"
        dst = selected_dir / f"panel{pid}.png"
        shutil.copy2(winner_path, dst)
        info(f"Panel {pid} → variant {winner_v} ({dst.name})")

    return winners


if __name__ == "__main__":
    main()
