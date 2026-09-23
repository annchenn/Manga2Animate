import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.log import init_log_file, rule, section, success, info, warn
from src.run_config import CONFIG_FILENAME, collect_models, write_run_config
from src.runs import latest_run_dir, new_run_id


GCP_PROJECT_MAP = {
    "1": os.getenv("GOOGLE_CLOUD_PROJECT"),
    "2": os.getenv("GOOGLE_CLOUD_PROJECT2"),
    "3": os.getenv("GOOGLE_CLOUD_PROJECT3"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the manga-to-storyboard pipeline.")
    parser.add_argument("-t", "--target", required=True, help="Target name under the target root")
    parser.add_argument(
        "--target-root",
        default="target",
        help="Directory holding the target cases (default: target)",
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Directory to write results into (default: results)",
    )
    parser.add_argument(
        "-n",
        "--num-iterations",
        type=int,
        default=None,
        help="Number of iterations to run, using the old behavior: -n 3 means iter0~iter2. Overridden by --to-iter.",
    )
    parser.add_argument(
        "--to-iter",
        type=int,
        default=None,
        help="Run until iter M, inclusive. Example: --to-iter 5 means run iter0~iter5.",
    )
    parser.add_argument(
        "--from-iter",
        type=int,
        default=None,
        help="Manually start from iter N. If omitted with --auto-resume, the first incomplete iter is detected automatically.",
    )
    parser.add_argument(
        "--time",
        default=None,
        help="Run folder timestamp (e.g. 20260821_143000) under <results-root>/<target>/ to write into "
             "or resume. Default: a new timestamp for a fresh run, the latest one when resuming.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Skip completed stages and continue from the first incomplete iter/stage.",
    )
    parser.add_argument(
        "-p",
        "--project",
        choices=["1", "2", "3"],
        default="1",
        help="Google Cloud project to use (1=GOOGLE_CLOUD_PROJECT, 2=GOOGLE_CLOUD_PROJECT2, 3=GOOGLE_CLOUD_PROJECT3); default 1",
    )
    parser.add_argument(
        "--storyboard-backend",
        choices=["gemini", "flux2"],
        default=None,
        help="Storyboard generation/evaluation backend (default: gemini, "
             "or STORYBOARD_BACKEND from .env if set)",
    )
    return parser.parse_args()


args = parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_RESULTS_DIR = BASE_DIR / args.results_root / args.target


def resolve_run_id() -> str:
    """Pick the run folder this invocation writes into.

    --time wins; otherwise a resuming run continues the newest existing folder and
    a fresh run starts a new timestamp, so runs never overwrite each other.
    """
    if args.time:
        return args.time
    if args.auto_resume or args.from_iter is not None:
        latest = latest_run_dir(TARGET_RESULTS_DIR)
        if latest is not None:
            return latest.name
    return new_run_id()


RUN_ID = resolve_run_id()
RUN_DIR = TARGET_RESULTS_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TARGET"] = args.target
os.environ["TARGET_ROOT"] = args.target_root
os.environ["RESULTS_ROOT"] = args.results_root
os.environ["RUN_ID"] = RUN_ID
os.environ["GOOGLE_CLOUD_PROJECT"] = GCP_PROJECT_MAP[args.project]

STORYBOARD_BACKEND = args.storyboard_backend or os.environ.get("STORYBOARD_BACKEND", "gemini")
if STORYBOARD_BACKEND not in {"gemini", "flux2"}:
    raise ValueError(f"Invalid storyboard backend '{STORYBOARD_BACKEND}' (must be gemini or flux2)")
os.environ["STORYBOARD_BACKEND"] = STORYBOARD_BACKEND

init_log_file(RUN_DIR / "log.txt")

# Import modules AFTER setting TARGET so module-level paths resolve correctly.
import src.magi as magi
import src.create_character_description as create_character_description
import src.create_script as create_script
import src.evaluate_scripts as evaluate_scripts
import src.generate_storyboard as generate_storyboard
import src.evaluate_storyboards as evaluate_storyboards


if args.to_iter is not None:
    if args.to_iter < 0:
        raise ValueError("--to-iter must be >= 0")
    FINAL_ITER = args.to_iter
    NUM_ITERATIONS = FINAL_ITER + 1
else:
    NUM_ITERATIONS = args.num_iterations if args.num_iterations is not None else int(
        os.environ.get("NUM_ITERATIONS", "3")
    )
    if NUM_ITERATIONS <= 0:
        raise ValueError("NUM_ITERATIONS / --num-iterations must be > 0")
    FINAL_ITER = NUM_ITERATIONS - 1

if args.from_iter is not None and not (0 <= args.from_iter <= FINAL_ITER):
    raise ValueError(f"--from-iter must be between 0 and {FINAL_ITER}")


# -------------------------
# Completion / checkpoint helpers
# -------------------------

def _nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _valid_json(path: Path) -> bool:
    if not _nonempty_file(path):
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except Exception:
        return False


def _first_existing(pattern: str, base: Path) -> Path | None:
    matches = sorted(base.glob(pattern))
    return matches[0] if matches else None


def read_mode_for_next_iter(prev_iter: int) -> str:
    meta_path = (
        evaluate_storyboards.RESULTS_DIR / f"iter{prev_iter}" / "intermediate" / "bestStoryboard_meta.json"
    )
    if not _valid_json(meta_path):
        raise FileNotFoundError(
            f"Cannot read mode for iter{prev_iter + 1}: {meta_path} is missing or invalid. "
            f"Please resume from iter{prev_iter} first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mode = meta.get("mode_for_next_iter", "one")
    if mode not in {"one", "two", "three"}:
        raise ValueError(f"Invalid mode '{mode}' in {meta_path}")
    return mode


def character_stage_done() -> bool:
    char_dir = create_character_description.RESULTS_DIR / "character"
    return _nonempty_file(char_dir / "transcript.txt") and _valid_json(
        char_dir / "character_description.json"
    )


def magi_stage_done() -> bool:
    return _nonempty_file(magi.RESULTS_DIR / "character" / "transcript.txt")


def character_description_done() -> bool:
    return _valid_json(create_character_description.RESULTS_DIR / "character" / "character_description.json")


def iter_dir(iteration: int) -> Path:
    return evaluate_storyboards.RESULTS_DIR / f"iter{iteration}"


def scripts_done(iteration: int) -> bool:
    scripts_dir = iter_dir(iteration) / "scripts"
    return all(_valid_json(scripts_dir / f"script_{i}.json") for i in range(1, create_script.NUM_SCRIPTS + 1))


def script_eval_done(iteration: int) -> bool:
    out_dir = iter_dir(iteration)
    interim = out_dir / "intermediate"
    required = [
        out_dir / "bestScript.json",
        interim / "feedback.json",
        interim / "character.json",
        *[interim / f"clip_{i}.json" for i in range(1, evaluate_scripts.NUM_CLIPS + 1)],
    ]
    return all(_valid_json(p) for p in required)


def storyboard_image_path(iteration: int, sb_index: int) -> Path | None:
    return _first_existing(f"storyboard_{sb_index}.*", iter_dir(iteration) / "storyboard")


def storyboards_done(iteration: int) -> bool:
    if STORYBOARD_BACKEND == "flux2":
        p = storyboard_image_path(iteration, 0)
        return p is not None and _nonempty_file(p)
    return all(
        (p := storyboard_image_path(iteration, i)) is not None and _nonempty_file(p)
        for i in range(generate_storyboard.NUM_STORYBOARDS)
    )


def best_storyboard_exists(iteration: int) -> bool:
    p = _first_existing("bestStoryboard.*", iter_dir(iteration))
    return p is not None and _nonempty_file(p)


def storyboard_eval_done(iteration: int) -> bool:
    interim = iter_dir(iteration) / "intermediate"
    return (
        best_storyboard_exists(iteration)
        and _valid_json(interim / "bestStoryboard_meta.json")
        and _valid_json(interim / "storyboard_feedback.json")
    )


def iteration_done(iteration: int) -> bool:
    return (
        scripts_done(iteration)
        and script_eval_done(iteration)
        and storyboards_done(iteration)
        and storyboard_eval_done(iteration)
    )


def first_incomplete_iter(final_iter: int) -> int:
    """Return the first iter in 0..final_iter that is not fully complete.
    If all are complete, return final_iter + 1.
    """
    for it in range(final_iter + 1):
        if not iteration_done(it):
            return it
    return final_iter + 1


def record_run_config() -> None:
    """Write the run's model config. Never fatal — a failed record is not worth
    losing the run over."""
    try:
        path, previous = write_run_config(RUN_DIR, RUN_ID, args.target)
        models = collect_models()
        info(
            f"Config: {path.relative_to(BASE_DIR)}  "
            f"(text={models['text']}, image={models['image']}, backend={models['storyboard_backend']})"
        )
        if previous is not None and previous != models:
            warn(
                f"{CONFIG_FILENAME} already recorded different models for {RUN_ID}: {previous}. "
                f"Results in this folder are a mix of both."
            )
    except Exception as e:
        warn(f"Could not write run config: {e}")


def run_stage(stage_name: str, done_fn, action):
    if args.auto_resume and done_fn():
        success(f"{stage_name} already complete — skipping")
        return

    section(stage_name)
    action()

    if args.auto_resume and not done_fn():
        raise RuntimeError(
            f"{stage_name} finished, but required output files are still missing or invalid. "
            f"Check the log before resuming."
        )


def run_iteration(iteration: int, mode: str):
    run_stage(
        f"ITER {iteration}  ·  create_script mode = {mode}",
        lambda: scripts_done(iteration),
        lambda: create_script.main(iteration=iteration, mode=mode),
    )
    run_stage(
        f"ITER {iteration}  ·  evaluate_scripts",
        lambda: script_eval_done(iteration),
        lambda: evaluate_scripts.main(iteration=iteration),
    )
    run_stage(
        f"ITER {iteration}  ·  generate_storyboard",
        lambda: storyboards_done(iteration),
        lambda: generate_storyboard.main(iteration=iteration),
    )
    run_stage(
        f"ITER {iteration}  ·  evaluate_storyboards",
        lambda: storyboard_eval_done(iteration),
        lambda: evaluate_storyboards.main(iteration=iteration),
    )


def main():
    rule(
        f"[bold cyan]Manga → Storyboard pipeline  ·  "
        f"target=[white]{args.target}[/]  ·  run=[white]{RUN_ID}[/]  ·  "
        f"final_iter=[white]{FINAL_ITER}[/]  ·  auto_resume=[white]{args.auto_resume}[/]  ·  "
        f"storyboard_backend=[white]{STORYBOARD_BACKEND}[/]"
    )

    info(f"Run directory: {RUN_DIR}")

    record_run_config()

    run_stage("STAGE 0  ·  MAGI transcript", magi_stage_done, magi.main)
    run_stage(
        "STAGE 0  ·  Character description",
        character_description_done,
        create_character_description.main,
    )

    if args.from_iter is not None:
        start_iter = args.from_iter
        info(f"Manual start: iter {start_iter}")
    elif args.auto_resume:
        start_iter = first_incomplete_iter(FINAL_ITER)
        if start_iter <= FINAL_ITER:
            info(f"Auto-resume detected first incomplete iter: iter {start_iter}")
        else:
            success(f"All iterations iter0~iter{FINAL_ITER} are already complete.")
            return
    else:
        start_iter = 0

    for it in range(start_iter, FINAL_ITER + 1):
        mode = "without" if it == 0 else read_mode_for_next_iter(it - 1)
        run_iteration(it, mode)

    section("Pipeline complete")
    success(f"All iteration(s) through iter{FINAL_ITER} finished.")


if __name__ == "__main__":
    main()
