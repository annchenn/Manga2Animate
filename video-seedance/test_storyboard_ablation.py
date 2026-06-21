"""
Ablation: character ref + manga pages + script, no storyboard, via Seedance.

Compares against create_video_fullstoryboard.py which uses bestStoryboard as input.
Looks for: ref.{jpg,jpeg,png} and manga_page, manga_page1, manga_page2, ... in target/<target>/

Result saved to: test/<target>/ablation_no_storyboard.mp4
"""
import os
import json
import time
import base64
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ARK_API_KEY   = os.getenv("ARK_API_KEY")
IMGBB_API_KEY = os.getenv("IMGBB_API_KEY")

SEEDANCE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
MODEL        = "dreamina-seedance-2-0-260128"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE  = PROJECT_ROOT / "prompts" / "video_generation_no_storyboard.txt"


def upload_imgbb(path):
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  uploading {Path(path).name} ({size_mb:.2f} MB)...")
    with open(path, "rb") as f:
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
    print(f"  → {url}")
    return url


def find_image(directory, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(directory) / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"No image for '{stem}' in {directory}")


def collect_manga_pages(target_dir):
    pages = []
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(target_dir) / f"manga_page{ext}"
        if p.exists():
            pages.append(p)
            break
    for i in range(1, 20):
        for ext in (".jpg", ".jpeg", ".png"):
            p = Path(target_dir) / f"manga_page{i}{ext}"
            if p.exists():
                pages.append(p)
                break
        else:
            break
    if not pages:
        raise FileNotFoundError(f"No manga pages found in {target_dir}")
    return pages


def submit(prompt, ref_url, page_urls, duration=15):
    content = [{"type": "text", "text": prompt}]
    content.append({"type": "image_url", "image_url": {"url": ref_url}, "role": "reference_image"})
    for url in page_urls:
        content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})

    payload = {
        "model": MODEL,
        "content": content,
        "ratio": "16:9",
        "resolution": "720p",
        "duration": duration,
        "generate_audio": True,
    }
    res = requests.post(SEEDANCE_URL, headers={
        "Authorization": f"Bearer {ARK_API_KEY}",
        "Content-Type": "application/json",
    }, json=payload)
    res.raise_for_status()
    return res.json()["id"]


def poll(task_id, interval=15, timeout=1800):
    headers = {"Authorization": f"Bearer {ARK_API_KEY}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = requests.get(f"{SEEDANCE_URL}/{task_id}", headers=headers).json()
        status = data.get("status")
        print(f"  [{task_id[-8:]}] {status}")
        if status == "succeeded":
            return data["content"]["video_url"]
        if status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"Task failed: {data}")
        time.sleep(interval)
    raise TimeoutError("Polling timed out")


def run_test(name, prompt, ref_url, page_urls, out_dir, duration):
    video_path = out_dir / f"{name}.mp4"
    task_path  = out_dir / f"{name}_task.json"

    if video_path.exists():
        print(f"  [skip] {name}.mp4 already exists")
        return

    if task_path.exists():
        task_id = json.loads(task_path.read_text())["task_id"]
        print(f"  [resume] polling {task_id}")
    else:
        task_id = submit(prompt, ref_url, page_urls, duration)
        task_path.write_text(json.dumps({"task_id": task_id}, indent=2))
        print(f"  submitted: {task_id}")

    url = poll(task_id)
    video_path.write_bytes(requests.get(url).content)
    print(f"  saved → {video_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-t", "--target", required=True)
    p.add_argument("--duration", type=int, default=15)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    target_dir  = PROJECT_ROOT / "target" / args.target
    script_path = PROJECT_ROOT / "results" / args.target / "iter1" / "bestScript.json"
    out_dir     = PROJECT_ROOT / "storyboard_ablation_test" / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = find_image(target_dir, "ref")
    pages    = collect_manga_pages(target_dir)
    script   = json.loads(script_path.read_text(encoding="utf-8"))

    duration = max(4, min(15, args.duration))

    prompt_template = PROMPT_FILE.read_text(encoding="utf-8")
    prompt = prompt_template.replace("<script>", json.dumps(script, ensure_ascii=False))

    print(f"target:  {args.target}")
    print(f"ref:     {ref_path.name}")
    print(f"pages:   {[p.name for p in pages]}")
    print(f"output:  {out_dir}")
    print()

    print("=== uploading images ===")
    ref_url   = upload_imgbb(str(ref_path))
    page_urls = [upload_imgbb(str(p)) for p in pages]

    print("\n=== ablation: ref + pages + script (no storyboard) ===")
    run_test("ablation_no_storyboard", prompt, ref_url, page_urls, out_dir, duration)

    print(f"\ndone → {out_dir}")
