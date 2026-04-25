import argparse
import json
import os
from pathlib import Path

from src.log import console, section, success


def parse_args():
    parser = argparse.ArgumentParser(description="Run the manga-to-storyboard pipeline.")
    parser.add_argument("-t", "--target", required=True, help="Target name under target/")
    parser.add_argument(
        "-n", "--num-iterations", type=int, default=None,
        help="Number of iterations (overrides NUM_ITERATIONS env var; default 3)",
    )
    return parser.parse_args()


args = parse_args()
os.environ["TARGET"] = args.target

# Import modules AFTER setting TARGET so module-level paths resolve correctly.
import src.magi as magi
import src.create_character_description as create_character_description
import src.create_script as create_script
import src.evaluate_scripts as evaluate_scripts
import src.generate_storyboard as generate_storyboard
import src.evaluate_storyboards as evaluate_storyboards


NUM_ITERATIONS = args.num_iterations if args.num_iterations is not None else int(
    os.environ.get("NUM_ITERATIONS", "3")
)


def read_mode_for_next_iter(prev_iter: int) -> str:
    meta_path = (
        evaluate_storyboards.RESULTS_DIR / f"iter{prev_iter}" / "intermediate" / "bestStoryboard_meta.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mode = meta.get("mode_for_next_iter", "one")
    if mode not in {"one", "two", "three"}:
        raise ValueError(f"Invalid mode '{mode}' in {meta_path}")
    return mode


def run_iteration(iteration: int, mode: str):
    create_script.main(iteration=iteration, mode=mode)
    evaluate_scripts.main(iteration=iteration)
    generate_storyboard.main(iteration=iteration)
    evaluate_storyboards.main(iteration=iteration)


def main():
    console.rule(
        f"[bold cyan]Manga → Storyboard pipeline  ·  "
        f"target=[white]{args.target}[/]  ·  iterations=[white]{NUM_ITERATIONS}[/]"
    )

    section("STAGE 0  ·  MAGI transcript")
    magi.main()

    section("STAGE 0  ·  Character description")
    create_character_description.main()

    section("ITER 0")
    run_iteration(0, "without")

    for it in range(1, NUM_ITERATIONS):
        mode = read_mode_for_next_iter(it - 1)
        section(f"ITER {it}  ·  create_script mode = [magenta]{mode}[/]")
        run_iteration(it, mode)

    section("Pipeline complete")
    success(f"All {NUM_ITERATIONS} iteration(s) finished.")


if __name__ == "__main__":
    main()
