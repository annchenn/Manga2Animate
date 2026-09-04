"""Storyboard → video, for every supported model, on the route that model can run.

Two routes, picked from the model rather than from a flag:

  ref route  (`--model seedance | minimax-local | minimax-api`)
      The whole `bestStoryboard.png` goes in as one reference image and the model
      returns one continuous video for the entire story. Two candidates, one
      Gemini review (`video_eval_first.txt`), best one wins.

  i2v route  (`--model wan | anisora`)
      The model only accepts a first/conditioning frame, so the storyboard is cut
      into panels (`crop_storyboard_panels.py`) and each panel drives its own
      clip. Two candidates per panel; panel 1 is reviewed with
      `video_eval_first.txt`, panels 2..N with `video_eval_sequential.txt`, which
      also sees the previous panel's winner and weights coherency at 0.4. The
      winners are concatenated into the final video.

Both routes read from  <results-root>/<target>/[<run>/]iter<N>/  — iter1 by
default, `--iter` to change it.

The per-model API details live in `video_models.py`; adding a model means adding
an entry there, not editing this file. `create_video_single.py` and
`create_video_fullstoryboard.py` are the original single-route scripts, kept as
backups.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from src import log
from src.runs import resolve_run_dir
from src.video_models import (
    BACKEND_NAMES,
    PolicyViolationError,
    fetch_result,
    get_backend,
    model_from_env,
    upload_image_to_imgbb,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")
GEMINI_REVIEW_MODEL = os.getenv("VIDEO_REVIEW_MODEL", "gemini-3.1-pro-preview")

PROMPT_PANEL         = PROJECT_ROOT / "prompts" / "video_generation.txt"
PROMPT_STORYBOARD    = PROJECT_ROOT / "prompts" / "video_generation_storyboard.txt"
EVAL_FIRST_FILE      = PROJECT_ROOT / "prompts" / "video_eval_first.txt"
EVAL_SEQUENTIAL_FILE = PROJECT_ROOT / "prompts" / "video_eval_sequential.txt"

IMG_EXTS = ("png", "jpg", "jpeg", "webp")
# anisora's tested runs used --base_seed 4096, so candidate 1 reproduces them
# exactly and further candidates step off from there. Ignored by the hosted
# providers, which never took a seed.
BASE_SEED = 4096


# ======================
# UTILITIES
# ======================
def save_file(content, folder, filename, is_json=False):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    mode = "w" if not isinstance(content, bytes) else "wb"
    encoding = None if isinstance(content, bytes) else "utf-8"
    with open(path, mode, encoding=encoding) as f:
        if is_json:
            json.dump(content, f, indent=4, ensure_ascii=False)
        else:
            f.write(content)
    return path


def read_state(state_path):
    """The task-id checkpoint next to a candidate. Missing or corrupt reads as empty."""
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(state_path, state):
    path = Path(state_path)
    save_file(state, str(path.parent), path.name, True)


def extract_json_from_response(text):
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return json.loads(text)


def get_panel_data(full_json, clip_index):
    """One clip plus only the characters that clip actually references."""
    clip = next((c for c in full_json.get("clips", []) if c["id"] == clip_index), None)
    if not clip:
        raise ValueError(f"Clip id {clip_index} not found in the script.")
    desc = clip.get("description", "")
    char_ids = set(int(x) for x in re.findall(r"Character\[(\d+)\]", desc))
    filtered_chars = [c for c in full_json.get("characters", []) if c["id"] in char_ids]
    return {"characters": filtered_chars, "clip": clip}


def calculate_optimal_duration(panel_image_path, total_panels_area, clip_data, clamp):
    """Longer clips for panels that are visually bigger and carry more dialogue."""
    try:
        with Image.open(panel_image_path) as panel_img:
            panel_area = panel_img.width * panel_img.height
        area_ratio = panel_area / total_panels_area if total_panels_area else 0.2
        visual_bonus = area_ratio * 10.0
    except Exception as e:
        log.warn(f"Area calculation failed ({e}), using default 2.0s")
        visual_bonus = 2.0
        area_ratio = 0

    # flux2-backend scripts carry a structured "dialogue" array; gemini-backend
    # ones embed "Dialogue(Character[N]): '...'" inside "description".
    dialogue_field = clip_data.get("dialogue")
    if isinstance(dialogue_field, list) and dialogue_field:
        total_words = sum(len(str(d.get("text", "")).split()) for d in dialogue_field if isinstance(d, dict))
    else:
        dialogues = re.findall(r"Dialogue\([^)]+\):\s*'([^']+)'", clip_data.get("description", ""))
        total_words = sum(len(d.split()) for d in dialogues)
    text_bonus = total_words / 10.0

    total = visual_bonus + text_bonus + 1.0
    lo, hi = clamp
    optimal = max(lo, min(hi, int(round(total))))
    log.info(
        f"Duration: visual({area_ratio*100:.1f}%)={visual_bonus:.2f}s"
        f" + dialogue({total_words}w)={text_bonus:.2f}s + base=1.00s → [bold]{optimal}s[/bold]"
        f"  [dim](clamp {lo}-{hi})[/dim]"
    )
    return optimal


# ======================
# GEMINI REVIEW
# ======================
class Reviewer:
    """Gemini video comparison.

    Two transports: the Files API (developer key) uploads each video and is the
    only one that works without Vertex ADC, which is why it is preferred here;
    the Vertex client has no Files API and gets the videos inline as bytes.
    """

    def __init__(self):
        from google import genai
        self.genai = genai
        if GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.transport = "files"
        elif os.getenv("GOOGLE_CLOUD_PROJECT"):
            self.client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
            self.transport = "inline"
        else:
            raise RuntimeError("Video review needs GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT")

    def _upload(self, path):
        log.step(f"Uploading to Gemini: {os.path.basename(path)}")
        f = self.client.files.upload(file=path)
        while f.state.name == "PROCESSING":
            time.sleep(5)
            f = self.client.files.get(name=f.name)
        if f.state.name == "FAILED":
            raise RuntimeError(f"Gemini file processing failed: {path}")
        return f

    def compare(self, user_prompt, video_a, video_b, prev_video=None):
        """Returns the raw response text. prev_video switches to the sequential
        rubric, which adds Coherency at weight 0.4."""
        template = EVAL_FIRST_FILE if prev_video is None else EVAL_SEQUENTIAL_FILE
        eval_prompt = template.read_text(encoding="utf-8").replace("<user_prompt>", user_prompt)

        if self.transport == "files":
            uploaded = []
            try:
                file_a = self._upload(video_a); uploaded.append(file_a)
                file_b = self._upload(video_b); uploaded.append(file_b)
                if prev_video is None:
                    contents = ["Video A:", file_a, "Video B:", file_b, eval_prompt]
                else:
                    file_prev = self._upload(prev_video); uploaded.append(file_prev)
                    contents = ["Reference Previous Video:", file_prev,
                                "Video A:", file_a, "Video B:", file_b, eval_prompt]
                log.step("Requesting Gemini evaluation...")
                resp = self.client.models.generate_content(
                    model=GEMINI_REVIEW_MODEL, contents=contents
                )
            finally:
                for f in uploaded:
                    try:
                        self.client.files.delete(name=f.name)
                    except Exception:
                        pass
            return resp.text.strip()

        from google.genai import types
        parts = []
        if prev_video is not None:
            parts += [types.Part.from_text(text="Reference Previous Video:"),
                      types.Part.from_bytes(data=Path(prev_video).read_bytes(), mime_type="video/mp4")]
        parts += [
            types.Part.from_text(text="Video A:"),
            types.Part.from_bytes(data=Path(video_a).read_bytes(), mime_type="video/mp4"),
            types.Part.from_text(text="Video B:"),
            types.Part.from_bytes(data=Path(video_b).read_bytes(), mime_type="video/mp4"),
            types.Part.from_text(text=eval_prompt),
        ]
        log.step("Requesting Gemini evaluation...")
        resp = self.client.models.generate_content(
            model=GEMINI_REVIEW_MODEL,
            contents=[types.Content(role="user", parts=parts)],
        )
        return resp.text.strip()


def pick_winner(reviewer, prompt, candidates, out_dir, prev_best=None):
    """Review candidates and return the winning path.

    One candidate means nothing to compare, so the review is skipped entirely —
    that is the normal case for local models, where a second candidate costs
    another full GPU run.
    """
    if len(candidates) == 1:
        return candidates[0]

    log.subsection("Gemini Evaluation")
    try:
        text = reviewer.compare(prompt, candidates[0], candidates[1], prev_best)
        save_file(text, out_dir, "gemini_eval_response.txt")
        decision = extract_json_from_response(text).get("Decision", "A_BETTER")
    except Exception as e:
        log.warn(f"Evaluation failed, defaulting to A. Error: {e}")
        decision = "A_BETTER"

    winner = "B" if decision == "B_BETTER" else "A"
    log.success(f"Decision: {decision} → using candidate {winner}")
    return candidates[1] if decision == "B_BETTER" else candidates[0]


# ======================
# GENERATION DRIVER
# ======================
def submit_candidate(backend, out_path, prompt, image, duration, seed, state_path, slot,
                     dry_run=False):
    """Submit one candidate and checkpoint its task id.

    Returns the task handle, or None when there is nothing to wait for — the
    video is already downloaded, or this is a dry run. Task ids are checkpointed
    next to the video so an interrupted run polls the existing job instead of
    paying for a second one; backends whose handle is not a string simply don't
    checkpoint.
    """
    out_path = Path(out_path)
    if out_path.exists():
        log.info(f"{out_path.name} already exists — skipping")
        return None

    # Check before submitting, and before a dry run reports success: a prompt over
    # the provider's limit is rejected at the door, and the point of --dry-run is to
    # find that out without spending anything.
    if backend.prompt_limit and len(prompt) > backend.prompt_limit:
        raise RuntimeError(
            f"{backend.name}: prompt is {len(prompt)} chars, over its {backend.prompt_limit}-char "
            f"limit. That is the hosted API's limit, not the model's — the self-hosted backend "
            f"has none, so `-m minimax-local` runs this prompt as it is."
        )

    if dry_run:
        log.warn(
            f"--dry-run: would submit {slot} to {backend.name} "
            f"({duration}s, seed {seed}, prompt {len(prompt)} chars, image {Path(image).name if not str(image).startswith('http') else image})"
        )
        return None

    state = read_state(state_path)
    if state.get("status") == "failed":
        raise PolicyViolationError(state.get("task_id"), state.get("message", "previous run failed"))

    task = state.get(slot)
    if task:
        log.info(f"Resuming {slot}: ...{str(task)[-8:]}")
        return task

    log.step(f"Submitting {slot} to {backend.name} ({duration}s)...")
    task = backend.submit(prompt, image, duration, seed=seed)
    if isinstance(task, str):
        state[slot] = task
        write_state(state_path, state)
    log.info(f"Submitted {slot}: ...{str(task)[-8:]}")
    return task


def collect_candidate(backend, out_path, task, state_path):
    """Poll a submitted task and save the video."""
    out_path = Path(out_path)
    if task is None:
        return out_path if out_path.exists() else None
    try:
        result = backend.poll(task)
    except PolicyViolationError as e:
        state = read_state(state_path)
        state.update({"status": "failed", "message": str(e), "task_id": e.task_id})
        write_state(state_path, state)
        raise
    fetch_result(result, out_path)
    log.success(f"{out_path.name} downloaded")
    return out_path


def generate_candidates(backend, specs, state_path, dry_run=False):
    """Every candidate is submitted before any is polled.

    The providers are asynchronous, so submitting both and then waiting lets them
    run at the same time; submitting one, waiting for it, then submitting the next
    doubles the wall clock for no benefit. This is what both original scripts did.
    """
    tasks = [
        submit_candidate(backend, s["out"], s["prompt"], s["image"], s["duration"],
                         s["seed"], state_path, s["slot"], dry_run)
        for s in specs
    ]
    if any(t is not None for t in tasks):
        log.step("Waiting for videos...")
    produced = [collect_candidate(backend, s["out"], t, state_path) for s, t in zip(specs, tasks)]
    return [p for p in produced if p is not None]


def candidate_specs(out_dir, prompt, image, duration, name="candidate"):
    """The two candidates every video is generated as.

    Two is the method, not a tuning knob: the reviewer's whole job is a pairwise
    A/B comparison, so one candidate would mean no review at all and results that
    cannot be compared with a model that did get reviewed.
    """
    return [
        {"out": Path(out_dir) / f"{name}_{c}.mp4", "prompt": prompt, "image": image,
         "duration": duration, "seed": BASE_SEED + c - 1, "slot": f"task_id_{c}"}
        for c in (1, 2)
    ]


# ======================
# ROUTE: ref  (one video from the whole storyboard)
# ======================
def run_ref_route(args, backend, paths, run_dir, reviewer):
    script = json.loads(paths["script"].read_text(encoding="utf-8"))
    prompt = PROMPT_STORYBOARD.read_text(encoding="utf-8").replace(
        "<script>", json.dumps(script, ensure_ascii=False)
    )
    save_file(prompt, run_dir, "prompt.txt")

    best_path = Path(run_dir) / "best_video.mp4"
    if best_path.exists():
        log.success("best_video.mp4 already exists — done")
        return best_path

    image = (upload_image_to_imgbb(str(paths["storyboard"]))
             if backend.needs_public_url and not args.dry_run else paths["storyboard"])

    candidates = generate_candidates(
        backend,
        candidate_specs(run_dir, prompt, image, backend.ref_duration),
        Path(run_dir) / "tasks_info.json",
        dry_run=args.dry_run,
    )
    if not candidates:
        log.warn("Nothing generated (dry run?) — stopping before review.")
        return None

    winner = pick_winner(reviewer, prompt, candidates, run_dir)
    shutil.copy(winner, best_path)
    return best_path


# ======================
# ROUTE: i2v  (one clip per storyboard panel)
# ======================
def build_panel_jobs(args, backend, paths, script):
    """Prompt, image and duration for each panel, in panel order."""
    panels_dir = paths["panels"]
    manga_dir = paths["manga_panels"]

    total_area = 0
    if manga_dir and manga_dir.is_dir():
        for i in range(1, args.panel_count + 1):
            p = manga_dir / f"panel{i}.png"
            if p.exists():
                with Image.open(p) as im:
                    total_area += im.width * im.height

    template = PROMPT_PANEL.read_text(encoding="utf-8")
    jobs = []
    for i in range(1, args.panel_count + 1):
        image = panels_dir / f"panel{i}.png"
        if not image.exists():
            log.warn(f"storyboard panel{i}.png not found, skipping.")
            continue
        try:
            panel_data = get_panel_data(script, i)
        except ValueError as e:
            log.warn(f"{e} — skipping panel {i}")
            continue

        manga_img = manga_dir / f"panel{i}.png" if manga_dir else None
        duration = calculate_optimal_duration(
            str(manga_img) if manga_img and manga_img.exists() else str(image),
            total_area, panel_data["clip"], backend.panel_clamp,
        )

        # A backend with a small text budget builds its own condensed prompt;
        # everything else gets the pipeline's per-panel template + the clip JSON.
        prompt = backend.build_panel_prompt(panel_data["clip"], panel_data["characters"])
        if prompt is None:
            prompt = template.replace("<script>", json.dumps(panel_data, ensure_ascii=False))

        jobs.append({"index": i, "prompt": prompt, "image": image, "duration": duration})
    return jobs


def run_i2v_route(args, backend, paths, run_dir, reviewer):
    run_dir = Path(run_dir)
    script = json.loads(paths["script"].read_text(encoding="utf-8"))
    jobs = build_panel_jobs(args, backend, paths, script)
    if not jobs:
        raise RuntimeError("No usable storyboard panels found.")

    # candidate_paths[c][panel index] -> mp4
    candidate_paths = {1: {}, 2: {}}

    if backend.batch:
        # One invocation per candidate: the model loads once and does every panel,
        # instead of paying the load cost per clip.
        for c in (1, 2):
            cand_dir = run_dir / f"candidate_{c}"
            existing = backend.collect_outputs(jobs, cand_dir) if cand_dir.is_dir() else []
            if existing and all(p is not None for p in existing):
                log.info(f"candidate_{c}: all {len(jobs)} clips exist — skipping generation")
                produced = existing
            else:
                log.section(f"Generating candidate {c} / 2 — {len(jobs)} panels")
                produced = backend.generate_batch(jobs, cand_dir, seed=BASE_SEED + c - 1)
            for job, path in zip(jobs, produced):
                if path is not None:
                    candidate_paths[c][job["index"]] = Path(path)
        if not candidate_paths[1]:
            log.warn("No clips produced (dry run?) — stopping before review.")
            return None
    else:
        for job in jobs:
            log.section(f"Panel {job['index']}")
            panel_dir = run_dir / f"panel_{job['index']}"
            panel_dir.mkdir(parents=True, exist_ok=True)
            image = (upload_image_to_imgbb(str(job["image"]))
                     if backend.needs_public_url and not args.dry_run else job["image"])
            specs = candidate_specs(panel_dir, job["prompt"], image, job["duration"])
            generate_candidates(backend, specs, panel_dir / "tasks_info.json", dry_run=args.dry_run)
            for c, spec in enumerate(specs, start=1):
                if spec["out"].exists():
                    candidate_paths[c][job["index"]] = spec["out"]

    if args.dry_run and not any(candidate_paths[1].values()):
        log.warn("--dry-run: nothing generated, stopping before review.")
        return None

    # Review each panel against the previous panel's winner (sequential rubric).
    best_videos = []
    prev_best = None
    for job in jobs:
        i = job["index"]
        panel_dir = run_dir / f"panel_{i}"
        panel_dir.mkdir(parents=True, exist_ok=True)
        best_path = panel_dir / f"best_panel_{i}.mp4"

        if best_path.exists():
            log.info(f"best_panel_{i}.mp4 already exists — skipping review")
        else:
            cands = [candidate_paths[c].get(i) for c in (1, 2)]
            cands = [c for c in cands if c is not None and Path(c).exists()]
            if not cands:
                log.warn(f"panel {i}: no candidate video, skipping")
                continue
            log.section(f"Reviewing panel {i}")
            winner = pick_winner(reviewer, job["prompt"], cands, panel_dir, prev_best)
            shutil.copy(winner, best_path)

        best_videos.append(best_path)
        prev_best = best_path

    if not best_videos:
        raise RuntimeError("No panel videos to concatenate.")

    log.section("Concatenating Clips")
    list_path = run_dir / "concat_list.txt"
    list_path.write_text(
        "".join(f"file '{p.resolve()}'\n" for p in best_videos), encoding="utf-8"
    )
    final_path = run_dir / "final_story.mp4"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-c", "copy", str(final_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        log.success(f"Final video: {final_path}")
    except Exception as e:
        log.error(f"ffmpeg concat failed: {e}")
    return final_path


# ======================
# PATHS
# ======================
def find_iter_dir(run_dir, iteration):
    """The requested iter, or the newest one that actually has a storyboard."""
    if iteration is not None:
        d = run_dir / f"iter{iteration}"
        if not d.is_dir():
            available = sorted(p.name for p in run_dir.glob("iter*") if p.is_dir())
            raise FileNotFoundError(
                f"iter{iteration} not found under {run_dir}. Available: "
                f"{', '.join(available) or 'none'} (--iter picks one, --iter -1 the newest)"
            )
        return d
    candidates = sorted(
        (p for p in run_dir.glob("iter*") if p.is_dir() and re.fullmatch(r"iter\d+", p.name)),
        key=lambda p: int(p.name[4:]),
        reverse=True,
    )
    for d in candidates:
        if any((d / f"bestStoryboard.{e}").exists() for e in IMG_EXTS):
            return d
    raise FileNotFoundError(f"No iter* with a bestStoryboard under {run_dir}")


def find_storyboard(iter_dir):
    for ext in IMG_EXTS:
        p = iter_dir / f"bestStoryboard.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"bestStoryboard not found in {iter_dir}")


def ensure_panels(args, iter_dir):
    """Cut the storyboard into panels if that hasn't been done yet.

    Cropping is a subprocess rather than an import because it pulls in MAGI's
    torch/transformers stack, which nothing else in this script needs.
    """
    panels_dir = iter_dir / "storyboard_panels"
    if panels_dir.is_dir() and any(panels_dir.glob("panel*.png")):
        return panels_dir

    log.section("Cropping storyboard panels")
    cmd = [
        sys.executable, str(PROJECT_ROOT / "src" / "crop_storyboard_panels.py"),
        "-t", args.target,
        "--results-root", str(PROJECT_ROOT / args.results_root),
        "--iter", iter_dir.name[len("iter"):],
    ]
    if args.time:
        cmd += ["--time", args.time]
    subprocess.run(cmd, check=True)

    if not (panels_dir.is_dir() and any(panels_dir.glob("panel*.png"))):
        raise FileNotFoundError(f"Panel cropping produced nothing in {panels_dir}")
    return panels_dir


def build_paths(args, backend):
    results_target = PROJECT_ROOT / args.results_root / args.target
    run_dir = resolve_run_dir(results_target, args.time)
    iter_dir = find_iter_dir(run_dir, args.iter)

    script = iter_dir / "bestScript.json"
    if not script.exists():
        raise FileNotFoundError(f"bestScript.json not found: {script}")

    manga_panels = PROJECT_ROOT / args.target_root / args.target / "manga_panels"
    paths = {
        "pipeline_run_dir": run_dir,
        "iter_dir": iter_dir,
        "storyboard": find_storyboard(iter_dir),
        "script": script,
        "manga_panels": manga_panels if manga_panels.is_dir() else None,
        "base_output_dir": results_target / "video" / args.model,
        "panels": None,
    }
    return paths


# ======================
# ARGS
# ======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Storyboard → video. The model decides the route: reference-image "
                    "models generate the whole story in one video, i2v models generate "
                    "one clip per storyboard panel."
    )
    parser.add_argument("-t", "--target", required=True, help="Target folder name, e.g. elf2")
    parser.add_argument("-m", "--model", default=None, choices=BACKEND_NAMES,
                        help="Video model. ref-route: seedance, minimax-local, minimax-api. "
                             "i2v-route: wan, anisora. Omit to detect it from "
                             "VIDEO_GENERATION_MODEL. The exact model id always comes "
                             "from that backend's own variable, e.g. SEEDANCE_MODEL.")
    parser.add_argument("--target-root", default="target", help="Directory holding the targets")
    parser.add_argument("--results-root", default="results", help="Directory holding the results")
    parser.add_argument("--time", default=None,
                        help="Pipeline run timestamp under <results-root>/<target>/ (default: newest)")
    parser.add_argument("--iter", type=int, default=1,
                        help="Iteration to read bestStoryboard/bestScript from (default: 1; "
                             "pass -1 for the newest iter that has a storyboard)")
    parser.add_argument("--panel-count", type=int, default=6, help="i2v route: number of panels")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Continue the newest run_* folder instead of starting a new one")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve paths, build every prompt and print what would be submitted, without calling any model (anisora also writes its prompt_list)")
    args = parser.parse_args()
    if args.iter is not None and args.iter < 0:
        args.iter = None
    return args


# ======================
# MAIN
# ======================
def main():
    args = parse_args()

    from_env = args.model is None
    if from_env:
        args.model = model_from_env()

    backend = get_backend(args.model)
    if hasattr(backend, "dry_run"):
        backend.dry_run = args.dry_run

    # The route is a property of the model, not a choice: a model either takes a
    # reference image or it does not.
    mode = backend.mode

    paths = build_paths(args, backend)

    log.section(f"Video Pipeline — {backend.name} ({mode} route) — {args.target}")
    if from_env:
        log.info("model        : from VIDEO_GENERATION_MODEL (no -m given)")
    log.info(f"pipeline run : {paths['pipeline_run_dir'].name}")
    log.info(f"iteration    : {paths['iter_dir'].name}")
    log.info(f"storyboard   : {paths['storyboard']}")
    log.info(f"script       : {paths['script']}")
    log.info(f"model id     : {backend.model_id}")
    log.info(f"review model : {GEMINI_REVIEW_MODEL}")

    # A dry run touches no provider, so it must not need their credentials —
    # inspecting the prompts is exactly what you do before you have the keys.
    if not args.dry_run:
        backend.check_env()

    if mode == "i2v":
        paths["panels"] = ensure_panels(args, paths["iter_dir"])
        log.info(f"panels       : {paths['panels']}")

    base_output_dir = paths["base_output_dir"]
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = None
    if args.auto_resume:
        runs = sorted(d for d in os.listdir(base_output_dir) if d.startswith("run_"))
        if runs:
            run_dir = base_output_dir / runs[-1]
            log.info(f"Resuming: {run_dir}")
    if run_dir is None:
        run_dir = base_output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"New run: {run_dir}")

    # A dry run never produces a video, so it must not require review credentials.
    reviewer = None if args.dry_run else Reviewer()

    try:
        if mode == "ref":
            result = run_ref_route(args, backend, paths, run_dir, reviewer)
        else:
            result = run_i2v_route(args, backend, paths, run_dir, reviewer)
    except PolicyViolationError as e:
        log.error(f"Policy violation — stopping: {e}")
        raise SystemExit(0)

    log.success(f"DONE → {result or run_dir}")


if __name__ == "__main__":
    main()
