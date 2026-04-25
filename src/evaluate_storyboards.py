import json
import mimetypes
import os
import re
import shutil
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.log import error, info, step, subsection, success, warn
from src.utils import with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
_target = os.environ.get("TARGET", "")
TARGET_DIR = BASE_DIR / "target" / _target if _target else BASE_DIR / "target"
RESULTS_DIR = BASE_DIR / "results" / _target if _target else BASE_DIR / "results"

PROMPT_DIR = BASE_DIR / "prompts"
SAME_PROMPT_FILE = PROMPT_DIR / "storyboard_reviewer_same_script.txt"
DIFF_PROMPT_FILE = PROMPT_DIR / "storyboard_reviewer_different_script.txt"
TIE_ORDER_FILE = PROMPT_DIR / "storyboard_tie_order.txt"

MODEL = os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview")
NUM_STORYBOARDS = int(os.environ.get("NUM_STORYBOARDS", "3"))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

CRITERIA_BY_NUMBER = {
    "1": "Character Identity Preservation",
    "2": "Overall Art Style Consistency",
    "3": "Color Presence & Visual Variety",
    "4": "Consistent Panel Size",
    "5": "Only Existing Characters",
    "6": "Narrative Order Consistency",
    "7": "Script Alignment Accuracy",
    "8": "Rich Background",
    "9": "Background Consistency",
    "10": "Character & Object Position Consistency",
    "11": "Object Layout Match",
    "12": "No Dialogue Boxes",
    "13": "Left/Right Feature Consistency",
}
NAME_TO_NUMBER = {v: k for k, v in CRITERIA_BY_NUMBER.items()}
INDEPENDENT_CRITERIA = {"3", "4"}

client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
)


# --- Path helpers ---

def iter_dir(iteration: int) -> Path:
    return RESULTS_DIR / f"iter{iteration}"


def find_image(directory: Path, stem: str) -> Path:
    matches = list(directory.glob(f"{stem}.*"))
    if not matches:
        raise FileNotFoundError(f"No image found for '{stem}' in {directory}")
    return matches[0]


def find_in_dir(directory: Path, stem: str) -> Path:
    matches = [p for p in directory.glob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not matches:
        raise FileNotFoundError(f"No image '{stem}.*' in {directory}")
    return matches[0]


def storyboard_path(iteration: int, sb_index: int) -> Path:
    return find_in_dir(iter_dir(iteration) / "storyboard", f"storyboard_{sb_index}")


def best_storyboard_path(iteration: int) -> Path:
    return find_in_dir(iter_dir(iteration), "bestStoryboard")


def part_from_image(path: Path) -> types.Part:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/png"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def strip_code_block(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


# --- Tie order parsing ---

def read_tie_order() -> list[str]:
    names = []
    for line in TIE_ORDER_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"\(\d+\)\s*(.+)", line)
        names.append(m.group(1).strip() if m else line)
    return names


# --- Reviewer calls ---

def call_reviewer(prompt: str, image_paths: list[Path], spinner_label: str = "Storyboard reviewer") -> dict:
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
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        return {"_raw": cleaned}
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return {"_raw": cleaned}


def compare_same_script(sb_a: Path, sb_b: Path, script_path: Path, out_path: Path) -> dict:
    template = SAME_PROMPT_FILE.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")
    prompt = template.replace("{Script JSON}", script_text)

    manga = find_image(TARGET_DIR, "manga_page")
    ref = find_image(TARGET_DIR, "ref")
    result = call_reviewer(prompt, [manga, ref, sb_a, sb_b], "same_script reviewer")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    success(f"Saved {out_path.name}")
    return result


def compare_different_script(
    sb_a: Path, script_a: Path,
    sb_b: Path, script_b: Path,
    out_path: Path,
) -> dict:
    template = DIFF_PROMPT_FILE.read_text(encoding="utf-8")
    prompt = template.replace("{Script 1 JSON}", script_a.read_text(encoding="utf-8"))
    prompt = prompt.replace("{Script 2 JSON}", script_b.read_text(encoding="utf-8"))

    manga = find_image(TARGET_DIR, "manga_page")
    ref = find_image(TARGET_DIR, "ref")
    result = call_reviewer(prompt, [manga, ref, sb_a, sb_b], "different_script reviewer")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    success(f"Saved {out_path.name}")
    return result


# --- Result parsing ---

def overall_winner(result: dict) -> str | None:
    """Returns 'A' or 'B' (or None if unparseable)."""
    label = result.get("overall_winner", {}).get("storyboard")
    if label in ("A", "B"):
        return label
    scores = result.get("scores", {})
    a, b = scores.get("A"), scores.get("B")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a > b:
            return "A"
        if b > a:
            return "B"
    return None


def is_pass(side_data) -> bool:
    if isinstance(side_data, dict):
        v = side_data.get("pass")
    else:
        v = side_data
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v == 1
    if isinstance(v, str):
        return v.strip().lower() in {"pass", "1", "true", "yes"}
    return False


def per_criterion_wins(result: dict, side: str) -> dict[str, int]:
    """For one comparison, return {criterion_number: 0|1} for the given side ('A' or 'B')."""
    out = {num: 0 for num in CRITERIA_BY_NUMBER}
    criteria = result.get("criteria_results", {})
    for num in CRITERIA_BY_NUMBER:
        crit = criteria.get(num, {})
        if num in INDEPENDENT_CRITERIA:
            if is_pass(crit.get(side)):
                out[num] = 1
        else:
            if crit.get("winner") == side:
                out[num] = 1
    return out


def winner_feedback(result: dict) -> dict:
    """Improvement suggestions block for the winner (already stored in result).
    Returns the dict (may be empty) — used as a 'storyboard feedback' entry."""
    return result.get("improvement_suggestions_for_winner", {}) or {}


# --- Tie breaking ---

def pairwise_break_tie(
    sb_indices: list[int],
    comparisons: dict[tuple[int, int], dict],
    tie_order: list[str],
) -> int:
    """Used when pairwise wins are tied (e.g. 1-1-1 in C(3,2)).
    Walks tie_order; for each criterion compares per-sb wins across all
    comparisons each sb participated in. First criterion with a unique
    leader wins."""
    sb_scores: dict[int, dict[str, int]] = {idx: {n: 0 for n in CRITERIA_BY_NUMBER} for idx in sb_indices}
    for (a_idx, b_idx), result in comparisons.items():
        if a_idx in sb_scores:
            for num, val in per_criterion_wins(result, "A").items():
                sb_scores[a_idx][num] += val
        if b_idx in sb_scores:
            for num, val in per_criterion_wins(result, "B").items():
                sb_scores[b_idx][num] += val

    for crit_name in tie_order:
        crit_num = NAME_TO_NUMBER.get(crit_name)
        if crit_num is None:
            continue
        per_sb = {idx: sb_scores[idx][crit_num] for idx in sb_indices}
        max_score = max(per_sb.values())
        leaders = [idx for idx, s in per_sb.items() if s == max_score]
        if len(leaders) == 1:
            info(f"Tie broken by '{crit_name}' (#{crit_num}): sb {leaders[0]} with {max_score} wins")
            return leaders[0]
    warn(f"Tie order exhausted; falling back to sb {sb_indices[0]}")
    return sb_indices[0]


# --- Pairwise same_script (used by iter 0 and 3-winner case in iter N+) ---

def run_pairwise_same_script(
    iteration: int,
    sb_indices: list[int],
    sb_paths: dict[int, Path],
    script_path: Path,
    interim_dir: Path,
) -> tuple[int, list[dict], str]:
    """Run C(N,2) same_script comparisons among sb_indices.
    Returns (winner_idx, feedback_list_for_winner, branch_suffix).
    branch_suffix in {'_clean', '_cyclic'}."""
    pairs: list[tuple[int, int]] = []
    n = len(sb_indices)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((sb_indices[i], sb_indices[j]))

    comparisons: dict[tuple[int, int], dict] = {}
    for a_idx, b_idx in pairs:
        out = interim_dir / f"sb_compare_same_{a_idx}_{b_idx}.json"
        step(f"same_script: sb {a_idx} vs sb {b_idx}")
        result = compare_same_script(
            sb_paths[a_idx], sb_paths[b_idx], script_path, out
        )
        comparisons[(a_idx, b_idx)] = result
        w = overall_winner(result)
        if w == "A":
            info(f"sb {a_idx} beats sb {b_idx}")
        elif w == "B":
            info(f"sb {b_idx} beats sb {a_idx}")
        else:
            info(f"sb {a_idx} vs sb {b_idx}: tie / unparseable")

    wins = {idx: 0 for idx in sb_indices}
    for (a_idx, b_idx), result in comparisons.items():
        w = overall_winner(result)
        if w == "A":
            wins[a_idx] += 1
        elif w == "B":
            wins[b_idx] += 1

    info(f"Pairwise win tally: {wins}")
    max_wins = max(wins.values())
    leaders = [idx for idx, w in wins.items() if w == max_wins]

    if len(leaders) == 1:
        winner_idx = leaders[0]
        suffix = "_clean"
    else:
        winner_idx = pairwise_break_tie(sb_indices, comparisons, read_tie_order())
        suffix = "_cyclic"

    feedbacks = []
    for (a_idx, b_idx), result in comparisons.items():
        if winner_idx not in (a_idx, b_idx):
            continue
        sb_label = "A" if winner_idx == a_idx else "B"
        if overall_winner(result) == sb_label:
            feedbacks.append(winner_feedback(result))

    return winner_idx, feedbacks, suffix


# --- Iter 0 ---

def run_iter0(iteration: int = 0):
    out_dir = iter_dir(iteration)
    interim = out_dir / "intermediate"
    interim.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / "bestScript.json"
    sb_indices = list(range(NUM_STORYBOARDS))
    sb_paths = {i: storyboard_path(iteration, i) for i in sb_indices}

    subsection(f"evaluate_storyboards (iter={iteration}, same_script pairwise)")
    winner_idx, feedbacks, branch_suffix = run_pairwise_same_script(
        iteration, sb_indices, sb_paths, script_path, interim
    )
    branch = f"iter0{branch_suffix}"

    src = sb_paths[winner_idx]
    dst = out_dir / f"bestStoryboard{src.suffix}"
    shutil.copy2(src, dst)
    success(f"Best storyboard: sb {winner_idx} → {dst.name}")

    mode = {1: "one", 2: "two"}.get(len(feedbacks), "one")
    write_meta_and_feedback(interim, branch, winner_idx, mode, feedbacks)


# --- Iter N+ ---

def run_iter_n(iteration: int):
    out_dir = iter_dir(iteration)
    interim = out_dir / "intermediate"
    interim.mkdir(parents=True, exist_ok=True)

    prev_dir = iter_dir(iteration - 1)
    script_path = out_dir / "bestScript.json"
    prev_script_path = prev_dir / "bestScript.json"
    prev_sb = best_storyboard_path(iteration - 1)

    sb_indices = list(range(NUM_STORYBOARDS))
    sb_paths = {i: storyboard_path(iteration, i) for i in sb_indices}

    subsection(f"evaluate_storyboards (iter={iteration}, vs prev best)")

    # 3 different_script comparisons: each new vs prev best (A=new, B=prev)
    diff_results: dict[int, dict] = {}
    diff_winners: list[int] = []
    for i in sb_indices:
        out = interim / f"sb_compare_diff_new{i}_vs_prev.json"
        step(f"different_script: new sb {i} vs prev best")
        result = compare_different_script(
            sb_paths[i], script_path, prev_sb, prev_script_path, out
        )
        diff_results[i] = result
        if overall_winner(result) == "A":
            diff_winners.append(i)
            info(f"new sb {i} beats prev best")
        else:
            info(f"prev best beats new sb {i}")

    n_winners = len(diff_winners)
    info(f"winners_count = {n_winners}: {diff_winners}")

    if n_winners == 0:
        # Carry forward
        branch = "iterN_carry"
        feedbacks = [winner_feedback(diff_results[i]) for i in sb_indices]
        shutil.copy2(prev_script_path, script_path)
        shutil.copy2(prev_dir / "intermediate" / "feedback.json",
                     interim / "feedback.json")
        dst = out_dir / f"bestStoryboard{prev_sb.suffix}"
        shutil.copy2(prev_sb, dst)
        warn(f"Carried forward bestScript / feedback / bestStoryboard from iter{iteration - 1}")
        write_meta_and_feedback(interim, branch, "carry", "three", feedbacks)
        return

    if n_winners == 1:
        winner_idx = diff_winners[0]
        branch = "iterN_1winner"
        feedbacks = [winner_feedback(diff_results[winner_idx])]
        src = sb_paths[winner_idx]
        dst = out_dir / f"bestStoryboard{src.suffix}"
        shutil.copy2(src, dst)
        success(f"Best storyboard: sb {winner_idx} → {dst.name}")
        write_meta_and_feedback(interim, branch, winner_idx, "one", feedbacks)
        return

    if n_winners == 2:
        a_idx, b_idx = diff_winners
        out = interim / f"sb_compare_same_{a_idx}_{b_idx}.json"
        step(f"same_script tiebreak: sb {a_idx} vs sb {b_idx}")
        result = compare_same_script(sb_paths[a_idx], sb_paths[b_idx], script_path, out)
        w = overall_winner(result)
        winner_idx = a_idx if w == "A" else b_idx
        loser_idx = b_idx if w == "A" else a_idx
        info(f"sb {winner_idx} beats sb {loser_idx}")
        same_feedback = winner_feedback(result)
        diff_feedback = winner_feedback(diff_results[winner_idx])
        feedbacks = [same_feedback, diff_feedback]
        branch = "iterN_2winner"
        src = sb_paths[winner_idx]
        dst = out_dir / f"bestStoryboard{src.suffix}"
        shutil.copy2(src, dst)
        success(f"Best storyboard: sb {winner_idx} → {dst.name}")
        write_meta_and_feedback(interim, branch, winner_idx, "two", feedbacks)
        return

    # n_winners == 3 → like iter 0 (same_script pairwise, ignore diff feedbacks)
    info("All 3 new beat prev best — running same_script pairwise (like iter 0)")
    winner_idx, feedbacks, branch_suffix = run_pairwise_same_script(
        iteration, sb_indices, sb_paths, script_path, interim
    )
    branch = f"iterN_3winner{branch_suffix}"
    src = sb_paths[winner_idx]
    dst = out_dir / f"bestStoryboard{src.suffix}"
    shutil.copy2(src, dst)
    success(f"Best storyboard: sb {winner_idx} → {dst.name}")
    mode = {1: "one", 2: "two"}.get(len(feedbacks), "one")
    write_meta_and_feedback(interim, branch, winner_idx, mode, feedbacks)


# --- Output writers ---

def write_meta_and_feedback(
    interim: Path,
    branch: str,
    winner_idx,
    mode_for_next_iter: str,
    feedbacks: list[dict],
):
    meta = {
        "branch": branch,
        "winner_idx": winner_idx,
        "mode_for_next_iter": mode_for_next_iter,
    }
    (interim / "bestStoryboard_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    success(f"Saved bestStoryboard_meta.json — branch={branch}, mode_for_next_iter={mode_for_next_iter}")

    (interim / "storyboard_feedback.json").write_text(
        json.dumps({"feedbacks": feedbacks}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    success(f"Saved storyboard_feedback.json ({len(feedbacks)} feedback(s))")


# --- Main ---

def main(iteration: int = 0):
    if iteration == 0:
        run_iter0(iteration)
    else:
        run_iter_n(iteration)


if __name__ == "__main__":
    main()
