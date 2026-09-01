import os
import re
import json
import time
import base64
import shutil
import argparse
import requests
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from google import genai
from google.genai import types

import log
from src.runs import resolve_run_dir

load_dotenv()


class PolicyViolationError(RuntimeError):
    def __init__(self, task_id, message):
        super().__init__(message)
        self.task_id = task_id
        self.message = message

# ======================
# CONFIG
# ======================
IMGBB_API_KEY         = os.getenv("IMGBB_API_KEY")
DASHSCOPE_API_KEY     = os.getenv("DASHSCOPE_API_KEY")
ARK_API_KEY           = os.getenv("ARK_API_KEY")
MINIMAX_API_KEY       = os.getenv("MINIMAX_API_KEY")
GOOGLE_CLOUD_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

VIDEO_GEN_MODEL     = os.getenv("VIDEO_GENERATION_MODEL", "dreamina-seedance-2-0-260128")
GEMINI_REVIEW_MODEL = os.getenv("VIDEO_REVIEW_MODEL", "gemini-3.1-pro-preview")

VIDEO_DURATION = 15  # fixed duration for full storyboard video

def _detect_model_type(model_str):
    m = model_str.lower()
    if "wan" in m:
        return "wan"
    if "minimax" in m or "hailuo" in m or m.startswith("h3"):
        return "minimax"
    return "seedance"

WAN_URL      = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
SEEDANCE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
MINIMAX_URL  = "https://api.minimax.io/v2/video_generation"
MINIMAX_QUERY_URL = "https://api.minimax.io/v2/query/video_generation"
# MiniMax-H3 caps the text prompt at 7000 characters.
MINIMAX_PROMPT_LIMIT = 7000

PROJECT_ROOT         = Path(__file__).resolve().parents[1]
PROMPT_FILE          = PROJECT_ROOT / "prompts" / "video_generation_storyboard.txt"
EVAL_FIRST_FILE      = PROJECT_ROOT / "prompts" / "video_eval_first.txt"

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


def upload_image_to_imgbb(image_path):
    file_size_mb = os.path.getsize(image_path) / 1024 / 1024
    log.step(f"Uploading to ImgBB ({file_size_mb:.2f} MB)...")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    res = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_API_KEY, "image": encoded},
        timeout=60,
    )
    res.raise_for_status()
    data = res.json()
    if not data.get("success"):
        raise RuntimeError(f"ImgBB upload failed: {data}")
    url = data["data"]["url"]
    log.info(f"ImgBB → {url}")
    return url

# ======================
# WAN API
# ======================
def submit_wan(prompt, image_url, duration):
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": VIDEO_GEN_MODEL,
        "input": {
            "prompt": prompt,
            "media": [{"type": "first_frame", "url": image_url}],
        },
        "parameters": {
            "size": "1K",
            "duration": duration,
            "prompt_extend": False,
            "watermark": False,
        },
    }
    res = requests.post(WAN_URL, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["output"]["task_id"]


def poll_wan(task_id, interval=15, timeout=1800):
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(
            f"https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}",
            headers=headers,
        ).json()
        status = res["output"]["task_status"]
        log.info(f"Task ...{task_id[-8:]} → {status}")
        if status == "SUCCEEDED":
            return res["output"]["video_url"]
        if status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"Task {task_id} failed: {res}")
        time.sleep(interval)
    raise TimeoutError(f"Polling timed out after {timeout}s")

# ======================
# SEEDANCE API
# ======================
def submit_seedance(prompt, image_url, duration):
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VIDEO_GEN_MODEL,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}, "role": "reference_image"},
        ],
        "ratio": "16:9",
        "resolution": "720p",
        "duration": duration,
        "generate_audio": True,
    }
    res = requests.post(SEEDANCE_URL, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["id"]


def poll_seedance(task_id, interval=15, timeout=1800):
    headers = {"Authorization": f"Bearer {ARK_API_KEY}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(f"{SEEDANCE_URL}/{task_id}", headers=headers)
        res.raise_for_status()
        data = res.json()
        status = data.get("status")
        log.info(f"Task ...{task_id[-8:]} → {status}")
        if status == "succeeded":
            return data["content"]["video_url"]
        if status in ("failed", "cancelled"):
            error = data.get("error", {})
            if "SensitiveContent" in error.get("code", "") or "PolicyViolation" in error.get("code", ""):
                raise PolicyViolationError(task_id, error.get("message", "Policy violation"))
            raise RuntimeError(f"Task {task_id} failed: {data}")
        if status == "expired":
            raise RuntimeError(f"Task {task_id} expired.")
        time.sleep(interval)
    raise TimeoutError(f"Polling timed out after {timeout}s")

# ======================
# MINIMAX API
# ======================
def submit_minimax(prompt, image_url, duration):
    if len(prompt) > MINIMAX_PROMPT_LIMIT:
        raise RuntimeError(
            f"Prompt is {len(prompt)} chars, over the MiniMax limit of {MINIMAX_PROMPT_LIMIT}"
        )
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VIDEO_GEN_MODEL,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}, "role": "reference_image"},
        ],
        "duration": duration,
        "resolution": "768P",
        "ratio": "16:9",
    }
    res = requests.post(MINIMAX_URL, headers=headers, json=payload)
    res.raise_for_status()
    data = res.json()
    task_id = data.get("task_id") or data.get("id")
    if not task_id:
        raise RuntimeError(f"MiniMax submission returned no task id: {data}")
    return task_id


def poll_minimax(task_id, interval=15, timeout=1800):
    headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(f"{MINIMAX_QUERY_URL}/{task_id}", headers=headers)
        res.raise_for_status()
        # v2 nests everything under "task": {id, status, error, content: {url}, ...}
        task = res.json().get("task", {})
        status = (task.get("status") or "").lower()
        log.info(f"Task ...{task_id[-8:]} → {status}")
        if status == "succeeded":
            return task["content"]["url"]
        if status in ("failed", "cancelled"):
            error = task.get("error", {}) or {}
            blob = f"{error.get('code', '')} {error.get('message', '')}".lower()
            if "sensitive" in blob or "policy" in blob or "violat" in blob:
                raise PolicyViolationError(task_id, error.get("message", "Policy violation"))
            raise RuntimeError(f"Task {task_id} failed: {task}")
        time.sleep(interval)
    raise TimeoutError(f"Polling timed out after {timeout}s")

# ======================
# MODEL DISPATCH
# ======================
def submit_task(model, prompt, image_url, duration):
    if model == "wan":
        return submit_wan(prompt, image_url, duration)
    if model == "minimax":
        return submit_minimax(prompt, image_url, duration)
    return submit_seedance(prompt, image_url, duration)


def poll_task(model, task_id):
    if model == "wan":
        return poll_wan(task_id)
    if model == "minimax":
        return poll_minimax(task_id)
    return poll_seedance(task_id)

# ======================
# GEMINI EVALUATION
# ======================
def evaluate_videos(client, user_prompt, video_a_path, video_b_path):
    eval_prompt = EVAL_FIRST_FILE.read_text(encoding="utf-8").replace("<user_prompt>", user_prompt)

    video_a_bytes = Path(video_a_path).read_bytes()
    video_b_bytes = Path(video_b_path).read_bytes()

    contents = [types.Content(role="user", parts=[
        types.Part.from_text(text="Video A:"),
        types.Part.from_bytes(data=video_a_bytes, mime_type="video/mp4"),
        types.Part.from_text(text="Video B:"),
        types.Part.from_bytes(data=video_b_bytes, mime_type="video/mp4"),
        types.Part.from_text(text=eval_prompt),
    ])]

    log.step("Requesting Gemini evaluation...")
    resp = client.models.generate_content(
        model=GEMINI_REVIEW_MODEL,
        contents=contents,
    )

    return resp.text.strip(), eval_prompt


def extract_json_from_response(text):
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return json.loads(text)

# ======================
# PATH HELPERS
# ======================
def find_latest_iter_dir(results_target_dir):
    if not results_target_dir.exists():
        raise FileNotFoundError(f"Results folder not found: {results_target_dir}")
    candidates = []
    for p in results_target_dir.iterdir():
        m = re.fullmatch(r"iter(\d+)", p.name)
        if m and p.is_dir():
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"No iter folder found under: {results_target_dir}")
    return max(candidates, key=lambda x: x[0])[1]


def find_storyboard(iter_dir):
    for ext in ("png", "jpg", "jpeg", "webp"):
        p = iter_dir / f"bestStoryboard.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"bestStoryboard not found in {iter_dir}")


def build_paths(args):
    results_target_dir = PROJECT_ROOT / args.results_root / args.target
    # --time picks which pipeline run to read the storyboard from; default: the newest.
    pipeline_run_dir   = resolve_run_dir(results_target_dir, args.time)
    latest_iter_dir    = find_latest_iter_dir(pipeline_run_dir)
    storyboard_path    = find_storyboard(latest_iter_dir)
    input_json_path    = latest_iter_dir / "bestScript.json"
    # Video output is not per-pipeline-run: <results-root>/<target>/video_fullstoryboard/<model>/run_*
    base_output_dir    = results_target_dir / "video_fullstoryboard" / args.model

    if not input_json_path.exists():
        raise FileNotFoundError(f"bestScript.json not found: {input_json_path}")

    return {
        "target_name":      args.target,
        "pipeline_run_dir": pipeline_run_dir,
        "storyboard_path":  storyboard_path,
        "input_json_path":  input_json_path,
        "base_output_dir":  base_output_dir,
    }

# ======================
# ARGS
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Full-storyboard video generation (single 15s video)")
    parser.add_argument("-t", "--target", required=True, help="Target folder name, e.g. lie_in_april1")
    parser.add_argument("--results-root", default="results",
                        help="Directory holding the results (default: results)")
    parser.add_argument("--time", default=None,
                        help="Pipeline run timestamp under <results-root>/<target>/ to take the storyboard "
                             "from (default: the latest run)")
    parser.add_argument("-m", "--model",  required=False, default=_detect_model_type(VIDEO_GEN_MODEL),
                        choices=["wan", "seedance", "minimax"], help="Video model to use")
    parser.add_argument("--auto-resume",  action="store_true",
                        help="Resume the latest run_* folder; skips generation if candidates exist")
    return parser.parse_args()

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    args  = parse_args()
    paths = build_paths(args)
    model = args.model

    target_name     = paths["target_name"]
    storyboard_path = paths["storyboard_path"]
    input_json_path = paths["input_json_path"]
    base_output_dir = paths["base_output_dir"]

    log.section(f"Full-Storyboard Video Pipeline — {model.upper()} — {target_name}")
    log.info(f"pipeline run     : {paths['pipeline_run_dir'].name}")
    log.info(f"storyboard       : {storyboard_path}")
    log.info(f"generation model : {VIDEO_GEN_MODEL}")
    log.info(f"review model     : {GEMINI_REVIEW_MODEL}")
    log.info(f"duration         : {VIDEO_DURATION}s")
    log.info(f"output           : {base_output_dir}")

    os.makedirs(base_output_dir, exist_ok=True)

    run_dir = None
    if args.auto_resume:
        runs = sorted(d for d in os.listdir(base_output_dir) if d.startswith("run_"))
        if runs:
            run_dir = os.path.join(base_output_dir, runs[-1])
            log.info(f"Resuming from: {run_dir}")

    if not run_dir:
        run_dir = os.path.join(base_output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        log.info(f"New run: {run_dir}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        full_json = json.load(f)

    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    prompt          = prompt_template.replace("<script>", json.dumps(full_json, ensure_ascii=False))
    gemini_client   = genai.Client(
        vertexai=True,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
    )

    best_video_path = os.path.join(run_dir, "best_video.mp4")

    if os.path.exists(best_video_path):
        log.success(f"best_video.mp4 already exists — done")
    else:
        video_1_path    = os.path.join(run_dir, "candidate_1.mp4")
        video_2_path    = os.path.join(run_dir, "candidate_2.mp4")
        tasks_info_path = os.path.join(run_dir, "tasks_info.json")

        if os.path.exists(video_1_path) and os.path.exists(video_2_path):
            log.info("Both candidates exist — skipping submission")
        else:
            if os.path.exists(tasks_info_path):
                with open(tasks_info_path, "r", encoding="utf-8") as f:
                    tasks_info = json.load(f)
                if tasks_info.get("status") == "failed":
                    log.error(f"Previous run failed with policy violation: {tasks_info.get('message')}")
                    log.error("Remove or fix the run folder before retrying.")
                    raise SystemExit(0)
                log.info("Found tasks_info.json — resuming poll")
                task_id_a = tasks_info["task_id_a"]
                task_id_b = tasks_info["task_id_b"]
            else:
                image_url = upload_image_to_imgbb(str(storyboard_path))

                log.step(f"Submitting candidates A & B to {model}...")
                task_id_a = submit_task(model, prompt, image_url, VIDEO_DURATION)
                task_id_b = submit_task(model, prompt, image_url, VIDEO_DURATION)
                save_file({"task_id_a": task_id_a, "task_id_b": task_id_b}, run_dir, "tasks_info.json", True)
                log.info(f"Submitted  A: ...{task_id_a[-8:]}  B: ...{task_id_b[-8:]}")

            log.step("Waiting for videos...")
            try:
                if not os.path.exists(video_1_path):
                    url_a = poll_task(model, task_id_a)
                    save_file(requests.get(url_a).content, run_dir, "candidate_1.mp4")
                if not os.path.exists(video_2_path):
                    url_b = poll_task(model, task_id_b)
                    save_file(requests.get(url_b).content, run_dir, "candidate_2.mp4")
            except PolicyViolationError as e:
                save_file(
                    {"task_id_a": task_id_a, "task_id_b": task_id_b,
                     "status": "failed", "message": str(e)},
                    run_dir, "tasks_info.json", True,
                )
                log.error(f"Policy violation — stopping pipeline: {e}")
                raise SystemExit(0)
            log.success("Both candidates downloaded")

        log.subsection("Gemini Evaluation")
        eval_vid_a = os.path.join(run_dir, "video_A.mp4")
        eval_vid_b = os.path.join(run_dir, "video_B.mp4")
        shutil.copy(video_1_path, eval_vid_a)
        shutil.copy(video_2_path, eval_vid_b)

        try:
            eval_text, _ = evaluate_videos(gemini_client, prompt, eval_vid_a, eval_vid_b)
            save_file(eval_text, run_dir, "gemini_eval_response.txt")
            eval_result = extract_json_from_response(eval_text)
            decision = eval_result.get("Decision", "A_BETTER")
        except Exception as e:
            log.warn(f"Evaluation failed, defaulting to A. Error: {e}")
            decision = "A_BETTER"

        winner = "B" if decision == "B_BETTER" else "A"
        log.success(f"Decision: {decision} → using candidate {winner}")

        best_src = video_2_path if decision == "B_BETTER" else video_1_path
        shutil.copy(best_src, best_video_path)

        if os.path.exists(eval_vid_a):
            os.remove(eval_vid_a)
        if os.path.exists(eval_vid_b):
            os.remove(eval_vid_b)

    log.success(f"DONE → {best_video_path}")
