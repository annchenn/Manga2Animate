"""Cross-check metrics.json's Gemini call count against the artifacts on disk.

The live counter in src/metrics.py is the easy number to trust and the easy number
to get wrong (a miswired stage, a client that bypassed the factory). This
reconstructs the count independently from what each stage actually left behind --
counting saved files, and reading the number of scored variants out of each panel
review rather than assuming FLUX2_NUM_VARIANTS -- and reports any disagreement.

It reconstructs `logical_calls` (pipeline steps), not `api_calls`: retries leave no
trace on disk, so artifacts can only ever account for the steps themselves.

    python opensource_backends/flux2/verify_metrics.py --run-dir results/<case>/<run>
    python opensource_backends/flux2/verify_metrics.py --target DualJustice1   # newest run

Generalised from manga2video/count_gemini_calls.py, which hard-coded one batch of
eight runs and their directory layout.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _count(path_glob) -> int:
    return len(sorted(path_glob))


def count_from_artifacts(run_dir: Path) -> tuple[dict, list[str]]:
    """Gemini steps implied by the files in `run_dir`, plus anything that looks off."""
    breakdown = {
        "character_description": 0,
        "script_generation": 0,
        "script_evaluation": 0,
        "panel_candidate_review": 0,
        "storyboard_generation": 0,
        "storyboard_comparison": 0,
    }
    notes: list[str] = []

    if (run_dir / "character" / "character_description.json").exists():
        breakdown["character_description"] = 1
    else:
        notes.append("no character/character_description.json")

    for iter_dir in sorted(run_dir.glob("iter*")):
        tag = iter_dir.name
        breakdown["script_generation"] += _count((iter_dir / "scripts").glob("script_*.json"))

        interim = iter_dir / "intermediate"
        # The script reviewer is one call for characters plus one per clip.
        if (interim / "character.json").exists():
            breakdown["script_evaluation"] += 1
        else:
            notes.append(f"{tag}: no intermediate/character.json")
        breakdown["script_evaluation"] += _count(interim.glob("clip_*.json"))

        # Panel reviews: read the actual number of scored variants rather than
        # assuming the configured variant count.
        for pr in sorted((iter_dir / "storyboard" / "candidates").glob("panel_review_*.json")):
            try:
                data = json.loads(pr.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                notes.append(f"{tag}: unreadable {pr.name}")
                continue
            breakdown["panel_candidate_review"] += len(data.get("per_variant", {}))

        # The gemini backend generates whole sheets with the image model; the flux2
        # backend draws them locally and makes no image call at all.
        breakdown["storyboard_generation"] += _count(
            (iter_dir / "storyboard").glob("storyboard_[0-9]*.png")
        ) if not (iter_dir / "storyboard" / "candidates").exists() else 0

        breakdown["storyboard_comparison"] += _count(interim.glob("sb_compare_*.json"))

    breakdown["_total"] = sum(v for k, v in breakdown.items() if not k.startswith("_"))
    return breakdown, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, help="results/<target>/<run_id>")
    ap.add_argument("--target", help="use this target's newest run under --results-root")
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    args = ap.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        if not args.target:
            ap.error("pass --run-dir or --target")
        runs = sorted(p for p in (args.results_root / args.target).glob("*") if p.is_dir())
        if not runs:
            print(f"no runs under {args.results_root / args.target}", file=sys.stderr)
            return 2
        run_dir = runs[-1]
    if not run_dir.is_dir():
        print(f"not a directory: {run_dir}", file=sys.stderr)
        return 2

    print(f"run: {run_dir}")
    breakdown, notes = count_from_artifacts(run_dir)

    print("\nGemini steps implied by artifacts:")
    for k, v in breakdown.items():
        if not k.startswith("_"):
            print(f"  {k:26} {v:4}")
    print(f"  {'TOTAL':26} {breakdown['_total']:4}")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"\nNo metrics.json in this run — nothing to compare against."
              f"\n(Artifact total above is still the ground-truth step count.)")
        for n in notes:
            print(f"  note: {n}")
        return 0

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    g = data.get("gemini", {})
    logical, api = g.get("logical_calls"), g.get("api_calls")
    print(f"\nmetrics.json: logical_calls={logical}  api_calls={api} "
          f"(retries={g.get('retries')}, failures={g.get('failures')})")
    print(f"backend: {data.get('storyboard_backend')}  wall_clock_s={data.get('wall_clock_s')}")

    ok = logical == breakdown["_total"]
    if ok:
        print(f"\nMATCH: {logical} logical call(s) both ways.")
    else:
        print(f"\nMISMATCH: metrics.json says {logical} logical call(s), artifacts imply "
              f"{breakdown['_total']}.")
        print("  Likely causes: a stage ran in an earlier run of the same folder "
              "(auto-resume reuses artifacts without recounting), a crash between a "
              "call and its file being written, or a client that bypassed "
              "src.utils.make_genai_client.")
        if data.get("stages_skipped"):
            print(f"  This run skipped: {', '.join(data['stages_skipped'])} "
                  f"— their artifacts exist but no calls were made for them, which "
                  f"explains an artifact count higher than the live one.")
    for n in notes:
        print(f"  note: {n}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
