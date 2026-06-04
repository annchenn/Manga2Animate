import os
import re
import json
import time
import base64
import shutil
import argparse
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from PIL import Image

import log

load_dotenv()

# ======================
# CONFIG
# ======================
IMGBB_API_KEY     = os.getenv("IMGBB_API_KEY")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") # for wan2.7 generation
ARK_API_KEY       = os.getenv("ARK_API_KEY") # for seedance generation
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")     # for video evaluation (Files API, developer only)
GCP_LOCATION      = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

VIDEO_GEN_MODEL     = os.getenv("VIDEO_GENERATION_MODEL", "dreamina-seedance-2-0-260128")
GEMINI_REVIEW_MODEL = os.getenv("VIDEO_REVIEW_MODEL", "gemini-3.1-pro-preview")

def _detect_model_type(model_str):
    return "wan" if "wan" in model_str.lower() else "seedance"

WAN_URL      = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
SEEDANCE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"

DURATION_CLAMP = {
    "wan":      {"min": 2, "max": 5},
    "seedance": {"min": 4, "max": 5},
}

PANEL_COUNT  = 6
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE  = PROJECT_ROOT / "prompts" / "video_generation.txt"
EVAL_FIRST_FILE      = PROJECT_ROOT / "prompts" / "video_eval_first.txt"
EVAL_SEQUENTIAL_FILE = PROJECT_ROOT / "prompts" / "video_eval_sequential.txt"

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


def get_panel_data(full_json, clip_index):
    clip = next((c for c in full_json.get("clips", []) if c["id"] == clip_index), None)
    if not clip:
        raise ValueError(f"Clip id {clip_index} not found in JSON.")
    desc = clip.get("description", "")
    char_ids = set(int(x) for x in re.findall(r"Character\[(\d+)\]", desc))
    filtered_chars = [c for c in full_json.get("characters", []) if c["id"] in char_ids]
    return {"characters": filtered_chars, "clip": clip}


def calculate_optimal_duration(panel_image_path, total_panels_area, clip_data, model):
    try:
        with Image.open(panel_image_path) as panel_img:
            panel_area = panel_img.width * panel_img.height
        area_ratio = panel_area / total_panels_area
        visual_bonus = area_ratio * 10.0
    except Exception as e:
        log.warn(f"Area calculation failed ({e}), using default 2.0s")
        visual_bonus = 2.0
        area_ratio = 0

    description = clip_data.get("description", "")
    dialogues = re.findall(r"Dialogue\([^)]+\):\s*'([^']+)'", description)
    total_words = sum(len(d.split()) for d in dialogues)
    text_bonus = total_words / 10.0

    base_time = 1.0
    total = visual_bonus + text_bonus + base_time

    clamp = DURATION_CLAMP[model]
    optimal = max(clamp["min"], min(clamp["max"], int(round(total))))
    log.info(
        f"Duration: visual({area_ratio*100:.1f}%)={visual_bonus:.2f}s"
        f" + dialogue({total_words}w)={text_bonus:.2f}s"
        f" + base={base_time:.2f}s → [bold]{optimal}s[/bold]"
        f"  [dim](clamp {clamp['min']}-{clamp['max']})[/dim]"
    )
    return optimal

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
            raise RuntimeError(f"Task {task_id} failed: {data}")
        if status == "expired":
            raise RuntimeError(f"Task {task_id} expired.")
        time.sleep(interval)
    raise TimeoutError(f"Polling timed out after {timeout}s")

# ======================
# MODEL DISPATCH
# ======================
def submit_task(model, prompt, image_url, duration):
    if model == "wan":
        return submit_wan(prompt, image_url, duration)
    return submit_seedance(prompt, image_url, duration)


def poll_task(model, task_id):
    if model == "wan":
        return poll_wan(task_id)
    return poll_seedance(task_id)

# ======================
# GEMINI EVALUATION
# ======================
def upload_to_gemini(client, file_path):
    log.step(f"Uploading to Gemini: {os.path.basename(file_path)}")
    gemini_file = client.files.upload(file=file_path)
    while gemini_file.state.name == "PROCESSING":
        log.info("Processing, waiting 5s...")
        time.sleep(5)
        gemini_file = client.files.get(name=gemini_file.name)
    if gemini_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini file processing failed: {file_path}")
    return gemini_file


def evaluate_videos(client, user_prompt, video_a_path, video_b_path, prev_video_path=None):
    is_first = (prev_video_path is None)

    if is_first:
        eval_prompt = EVAL_FIRST_FILE.read_text(encoding="utf-8").replace("<user_prompt>", user_prompt)
    else:
        eval_prompt = EVAL_SEQUENTIAL_FILE.read_text(encoding="utf-8").replace("<user_prompt>", user_prompt)

    file_a = upload_to_gemini(client, video_a_path)
    file_b = upload_to_gemini(client, video_b_path)

    if is_first:
        contents = ["Video A:", file_a, "Video B:", file_b, eval_prompt]
    else:
        file_prev = upload_to_gemini(client, prev_video_path)
        contents = ["Reference Previous Video:", file_prev, "Video A:", file_a, "Video B:", file_b, eval_prompt]

    log.step("Requesting Gemini evaluation...")
    resp = client.models.generate_content(
        model=GEMINI_REVIEW_MODEL,
        contents=contents,
    )

    client.files.delete(name=file_a.name)
    client.files.delete(name=file_b.name)
    if not is_first:
        client.files.delete(name=file_prev.name)

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


def build_paths(args):
    target_dir            = PROJECT_ROOT / "target" / args.target
    manga_panels_dir      = target_dir / "manga_panels"
    storyboard_panels_dir = target_dir / "storyboard_panels"
    results_target_dir    = PROJECT_ROOT / "results" / args.target
    latest_iter_dir       = find_latest_iter_dir(results_target_dir)
    input_json_path       = latest_iter_dir / "bestScript.json"
    base_output_dir       = results_target_dir / "video" / args.model

    for path, label in [
        (target_dir,            "target"),
        (manga_panels_dir,      "manga_panels"),
        (storyboard_panels_dir, "storyboard_panels"),
        (input_json_path,       "bestScript.json"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    return {
        "target_name":           args.target,
        "manga_panels_dir":      manga_panels_dir,
        "storyboard_panels_dir": storyboard_panels_dir,
        "input_json_path":       input_json_path,
        "base_output_dir":       base_output_dir,
    }

# ======================
# ARGS
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Manga-to-video pipeline (wan or seedance)")
    parser.add_argument("-t", "--target",    required=True, help="Target folder name, e.g. lie_in_april1")
    parser.add_argument("-m", "--model",     required=False, default=_detect_model_type(VIDEO_GEN_MODEL), choices=["wan", "seedance"], help="Video model to use (default: auto-detected from VIDEO_GENERATION_MODEL in .env)")
    parser.add_argument("--panel-count",     type=int, default=PANEL_COUNT)
    parser.add_argument("--auto-resume",     action="store_true",
                        help="Resume the latest run_* folder; skips generation if candidates exist, skips panel if best_panel exists")
    return parser.parse_args()

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    args  = parse_args()
    paths = build_paths(args)
    model = args.model

    target_name           = paths["target_name"]
    manga_panels_dir      = paths["manga_panels_dir"]
    storyboard_panels_dir = paths["storyboard_panels_dir"]
    input_json_path       = paths["input_json_path"]
    base_output_dir       = paths["base_output_dir"]

    log.section(f"Video Pipeline — {model.upper()} — {target_name}")
    log.info(f"generation model : {VIDEO_GEN_MODEL}")
    log.info(f"review model     : {GEMINI_REVIEW_MODEL}")
    log.info(f"script           : {input_json_path}")
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
    # Files API (used for video evaluation) is only supported in the Developer API, not Vertex AI
    gemini_client   = genai.Client(api_key=GEMINI_API_KEY)

    total_panels_area = sum(
        Image.open(manga_panels_dir / f"panel{i}.png").width *
        Image.open(manga_panels_dir / f"panel{i}.png").height
        for i in range(1, args.panel_count + 1)
        if (manga_panels_dir / f"panel{i}.png").exists()
    )

    best_videos          = []
    prev_best_video_path = None

    for i in range(1, args.panel_count + 1):
        log.section(f"Panel {i} / {args.panel_count}")

        panel_dir = os.path.join(run_dir, f"panel_{i}")
        os.makedirs(panel_dir, exist_ok=True)

        final_panel_video_path = os.path.join(panel_dir, f"best_panel_{i}.mp4")

        if os.path.exists(final_panel_video_path):
            log.info(f"best_panel_{i}.mp4 already exists — skipping")
            best_videos.append(final_panel_video_path)
            prev_best_video_path = final_panel_video_path
            continue

        local_img_path = storyboard_panels_dir / f"panel{i}.png"
        manga_img_path = manga_panels_dir / f"panel{i}.png"

        if not local_img_path.exists():
            log.warn(f"storyboard panel{i}.png not found, skipping.")
            continue
        if not manga_img_path.exists():
            log.warn(f"manga panel{i}.png not found, skipping.")
            continue

        panel_data       = get_panel_data(full_json, i)
        optimal_duration = calculate_optimal_duration(str(manga_img_path), total_panels_area, panel_data["clip"], model)
        prompt           = prompt_template.replace("<script>", json.dumps(panel_data, ensure_ascii=False))

        video_1_path    = os.path.join(panel_dir, "candidate_1.mp4")
        video_2_path    = os.path.join(panel_dir, "candidate_2.mp4")
        tasks_info_path = os.path.join(panel_dir, "tasks_info.json")

        if os.path.exists(video_1_path) and os.path.exists(video_2_path):
            log.info("Both candidates exist — skipping submission")
        else:
            if os.path.exists(tasks_info_path):
                log.info("Found tasks_info.json — resuming poll")
                with open(tasks_info_path, "r", encoding="utf-8") as f:
                    tasks_info = json.load(f)
                task_id_a = tasks_info["task_id_a"]
                task_id_b = tasks_info["task_id_b"]
            else:
                image_url = upload_image_to_imgbb(str(local_img_path))

                log.step(f"Submitting candidates A & B to {model}...")
                task_id_a = submit_task(model, prompt, image_url, optimal_duration)
                task_id_b = submit_task(model, prompt, image_url, optimal_duration)
                save_file({"task_id_a": task_id_a, "task_id_b": task_id_b}, panel_dir, "tasks_info.json", True)
                log.info(f"Submitted  A: ...{task_id_a[-8:]}  B: ...{task_id_b[-8:]}")

            log.step("Waiting for videos...")
            if not os.path.exists(video_1_path):
                url_a = poll_task(model, task_id_a)
                save_file(requests.get(url_a).content, panel_dir, "candidate_1.mp4")
            if not os.path.exists(video_2_path):
                url_b = poll_task(model, task_id_b)
                save_file(requests.get(url_b).content, panel_dir, "candidate_2.mp4")
            log.success("Both candidates downloaded")

        log.subsection("Gemini Evaluation")
        eval_vid_a = os.path.join(panel_dir, "video_A.mp4")
        eval_vid_b = os.path.join(panel_dir, "video_B.mp4")
        shutil.copy(video_1_path, eval_vid_a)
        shutil.copy(video_2_path, eval_vid_b)

        try:
            eval_text, eval_prompt_used = evaluate_videos(
                gemini_client, prompt, eval_vid_a, eval_vid_b, prev_best_video_path
            )
            save_file(eval_text, panel_dir, "gemini_eval_response.txt")
            eval_result = extract_json_from_response(eval_text)
            decision = eval_result.get("Decision", "A_BETTER")
        except Exception as e:
            log.warn(f"Evaluation failed, defaulting to A. Error: {e}")
            decision = "A_BETTER"

        winner = "B" if decision == "B_BETTER" else "A"
        log.success(f"Decision: {decision} → using candidate {winner}")

        best_src = video_2_path if decision == "B_BETTER" else video_1_path
        shutil.copy(best_src, final_panel_video_path)

        if os.path.exists(eval_vid_a):
            os.remove(eval_vid_a)
        if os.path.exists(eval_vid_b):
            os.remove(eval_vid_b)

        best_videos.append(final_panel_video_path)
        prev_best_video_path = final_panel_video_path

    log.section("Concatenating Clips")
    list_file_path = os.path.join(run_dir, "concat_list.txt")
    with open(list_file_path, "w", encoding="utf-8") as f:
        for vid in best_videos:
            f.write(f"file '{os.path.abspath(vid)}'\n")

    final_output_path = os.path.join(run_dir, "final_story.mp4")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file_path, "-c", "copy", final_output_path,
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log.success(f"Final video: {final_output_path}")
    except Exception as e:
        log.error(f"ffmpeg concat failed: {e}")

    log.success(f"DONE → {run_dir}")
