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

ARK_API_KEY   = os.getenv("ARK_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

SEEDANCE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
MODEL        = "dreamina-seedance-2-0-260128"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROMPT = """[Image 1] is a manga page. [Image 2] is the character reference sheet.
Animate the manga page into a short video. Bring the characters to life with natural motion based on the scene depicted.
Do not render any text in the video — eliminate all dialogue boxes, speech bubbles, and sound effect text.
"""

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

def submit_seedance(prompt, manga_url, ref_url, duration=10):
    headers = {
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": manga_url}, "role": "reference_image"},
            {"type": "image_url", "image_url": {"url": ref_url},   "role": "reference_image"},
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

def find_image(directory, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = directory / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"No image found for '{stem}' in {directory} (tried .jpg/.jpeg/.png)")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True, help="Target folder name, e.g. frieren1")
    parser.add_argument("--duration", type=int, default=15, help="Video duration in seconds (4-15, default 15)")
    parser.add_argument("--manga-page", default="manga_page", help="Manga page filename stem (no extension)")
    parser.add_argument("--ref", default="ref", help="Character ref filename stem (no extension)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    target_dir  = PROJECT_ROOT / "target" / args.target
    manga_path  = find_image(target_dir, args.manga_page)
    ref_path    = find_image(target_dir, args.ref)
    output_dir  = PROJECT_ROOT / "video-seedance" / "pipeline_generations_manga" / args.target

    os.makedirs(output_dir, exist_ok=True)
    run_dir = os.path.join(output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    print("start:", datetime.now())
    print(f"target:     {args.target}")
    print(f"manga:      {manga_path}")
    print(f"ref:        {ref_path}")
    print(f"output:     {run_dir}")

    task_info_path   = os.path.join(run_dir, "task_info.json")
    final_video_path = os.path.join(run_dir, "final_story.mp4")

    print("    Uploading manga page to ImgBB...")
    manga_url = upload_image_to_imgbb(str(manga_path))

    print("    Uploading character ref to ImgBB...")
    ref_url = upload_image_to_imgbb(str(ref_path))

    duration = max(4, min(15, args.duration))
    print(f"    Submitting task to Seedance (duration={duration}s)...")
    task_id = submit_seedance(PROMPT, manga_url, ref_url, duration=duration)
    with open(task_info_path, "w", encoding="utf-8") as f:
        json.dump({"task_id": task_id}, f, indent=4)
    print(f"    Submitted: {task_id}")

    print("    Waiting for video...")
    url = poll_seedance(task_id)
    with open(final_video_path, "wb") as f:
        f.write(requests.get(url).content)
    print(f"    Downloaded.")

    print(f"Done! Final video: {final_video_path}")
