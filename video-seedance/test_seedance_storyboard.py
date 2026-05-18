import os
import re
import json
import time
import base64
import argparse
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ======================
# CONFIG
# ======================
ARK_API_KEY   = os.getenv("ARK_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

SEEDANCE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
MODEL        = "dreamina-seedance-2-0-260128"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE  = PROJECT_ROOT / "prompts" / "video_generation_storyboard.txt"

# ======================
# SAVE TOOL
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

# ======================
# IMGBB UPLOAD
# ======================
def upload_image_to_imgbb(image_path):
    file_size_mb = os.path.getsize(image_path) / 1024 / 1024
    print(f"    Uploading to ImgBB... ({image_path}, {file_size_mb:.2f} MB)")
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
    print(f"    ImgBB → {url}")
    return url

# ======================
# SEEDANCE API
# ======================
def submit_seedance(prompt, image_url, duration=10):
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}, "role": "reference_image"},
        ],
        "ratio": "16:9",
        "resolution": "720p",
        "duration": duration,
        "generate_audio": False,
    }
    res = requests.post(SEEDANCE_URL, headers=headers, json=payload)
    res.raise_for_status()
    data = res.json()
    return data["id"]

def poll_seedance(task_id, interval=15, timeout=1800):
    headers = {"Authorization": f"Bearer {ARK_API_KEY}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(f"{SEEDANCE_URL}/{task_id}", headers=headers)
        res.raise_for_status()
        data = res.json()
        status = data.get("status")
        print(f"        Task {task_id[-8:]}... status: {status}")
        if status == "succeeded":
            return data["content"]["video_url"]
        if status in ["failed", "cancelled"]:
            raise RuntimeError(f"Task {task_id} failed: {data}")
        if status == "expired":
            raise RuntimeError(f"Task {task_id} expired.")
        time.sleep(interval)
    raise TimeoutError(f"Polling timed out after {timeout}s")

# ======================
# ARG PARSE & PATHS
# ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True, help="Target folder name, e.g. frieren1")
    parser.add_argument("--duration", type=int, default=15, help="Video duration in seconds (4-15, default 15)")
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()

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
    results_target_dir = PROJECT_ROOT / "results" / args.target
    latest_iter_dir    = find_latest_iter_dir(results_target_dir)
    input_json_path    = latest_iter_dir / "bestScript.json"
    storyboard_path    = latest_iter_dir / "bestStoryboard.png"
    base_output_dir    = PROJECT_ROOT / "video-seedance" / "pipeline_generations_storyboard" / args.target

    for path, label in [
        (input_json_path, "bestScript.json"),
        (storyboard_path, "bestStoryboard.png"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    return {
        "target_name":    args.target,
        "input_json_path": input_json_path,
        "storyboard_path": storyboard_path,
        "base_output_dir": base_output_dir,
    }

# ======================
# MAIN PIPELINE
# ======================
if __name__ == "__main__":
    args  = parse_args()
    paths = build_paths(args)

    target_name    = paths["target_name"]
    input_json_path = paths["input_json_path"]
    storyboard_path = paths["storyboard_path"]
    base_output_dir = paths["base_output_dir"]

    print("start:", datetime.now())
    print(f"target:     {target_name}")
    print(f"script:     {input_json_path}")
    print(f"storyboard: {storyboard_path}")
    print(f"output:     {base_output_dir}")

    os.makedirs(base_output_dir, exist_ok=True)

    run_dir = None
    if args.auto_resume:
        runs = sorted([d for d in os.listdir(base_output_dir) if d.startswith("run_")])
        if runs:
            run_dir = os.path.join(base_output_dir, runs[-1])
            print(f"[Resume] continuing from: {run_dir}")

    if not run_dir:
        run_dir = os.path.join(base_output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"[New run] {run_dir}")

    with open(input_json_path, "r", encoding="utf-8") as f:
        full_json = json.load(f)

    prompt = PROMPT_FILE.read_text(encoding="utf-8").replace("<script>", json.dumps(full_json, ensure_ascii=False))

    final_video_path = os.path.join(run_dir, "final_story.mp4")
    task_info_path   = os.path.join(run_dir, "task_info.json")

    if os.path.exists(final_video_path):
        print(f"[Resume] final_story.mp4 already exists, done.")
    else:
        if os.path.exists(task_info_path):
            print("    [Resume] found task_info.json, resuming poll...")
            with open(task_info_path, "r", encoding="utf-8") as f:
                task_info = json.load(f)
            task_id = task_info["task_id"]
        else:
            print("    Uploading storyboard to ImgBB...")
            image_url = upload_image_to_imgbb(str(storyboard_path))

            duration = max(4, min(15, args.duration))
            print(f"    Submitting task to Seedance (duration={duration}s)...")
            task_id = submit_seedance(prompt, image_url, duration=duration)
            save_file({"task_id": task_id}, run_dir, "task_info.json", True)
            print(f"    Submitted: {task_id}")

        print("    Waiting for video...")
        url = poll_seedance(task_id)
        save_file(requests.get(url).content, run_dir, "final_story.mp4")
        print(f"    Downloaded.")

    print(f"Done! Final video: {final_video_path}")
    print("DONE ->", run_dir)
