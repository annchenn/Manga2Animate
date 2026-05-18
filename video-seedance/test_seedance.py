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
# from google import genai  # used by Gemini comparison
from PIL import Image

load_dotenv()

# ======================
# CONFIG
# ======================
ARK_API_KEY    = os.getenv("ARK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMGBB_API_KEY  = os.getenv("IMGBB_API_KEY")

SEEDANCE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
MODEL        = "dreamina-seedance-2-0-260128"
PANEL_COUNT  = 6

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # comic2animate/
PROMPT_FILE  = PROJECT_ROOT / "prompts" / "video_generation.txt"

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
# JSON FILTERING
# ======================
def get_panel_data(full_json, clip_index):
    clip = next((c for c in full_json.get("clips", []) if c["id"] == clip_index), None)
    if not clip:
        raise ValueError(f"Clip with id {clip_index} not found in JSON.")
    desc = clip.get("description", "")
    char_ids = set([int(x) for x in re.findall(r"Character\[(\d+)\]", desc)])
    filtered_chars = [c for c in full_json.get("characters", []) if c["id"] in char_ids]
    return {"characters": filtered_chars, "clip": clip}

# ======================
# DYNAMIC DURATION
# ======================
def calculate_optimal_duration_int(panel_image_path, manga_page_path, clip_data):
    try:
        with Image.open(manga_page_path) as full_img:
            full_area = full_img.width * full_img.height
        with Image.open(panel_image_path) as panel_img:
            panel_area = panel_img.width * panel_img.height
        area_ratio = panel_area / full_area
        visual_bonus = area_ratio * 10.0
    except Exception as e:
        print(f"        Warning: area calculation failed ({e}), using default 2.0s")
        visual_bonus = 2.0
        area_ratio = 0

    description = clip_data.get("description", "")
    dialogues = re.findall(r"Dialogue\([^)]+\):\s*'([^']+)'", description)
    total_words = sum(len(d.split()) for d in dialogues)
    text_bonus = total_words / 10.0

    base_time = 1.0
    total_duration = visual_bonus + text_bonus + base_time
    optimal = max(4, min(10, int(round(total_duration))))  # Seedance supports 4-15s
    print(f"        Duration: visual({area_ratio*100:.1f}%):{visual_bonus:.2f}s + dialogue({total_words}w):{text_bonus:.2f}s + base:{base_time:.2f}s → {optimal}s")
    return optimal

# ======================
# SEEDANCE API
# ======================
def submit_seedance(prompt, image_url, duration=5):
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
# GEMINI EVALUATION
# ======================
def upload_to_gemini(client, file_path):
    print(f"    Uploading to Gemini: {os.path.basename(file_path)}...")
    gemini_file = client.files.upload(file=file_path)
    while gemini_file.state.name == "PROCESSING":
        print("    Processing, waiting 5s...")
        time.sleep(5)
        gemini_file = client.files.get(name=gemini_file.name)
    if gemini_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini file processing failed: {file_path}")
    return gemini_file

def evaluate_videos(client, user_prompt, video_a_path, video_b_path, prev_video_path=None):
    is_first = (prev_video_path is None)
    file_a = upload_to_gemini(client, video_a_path)
    file_b = upload_to_gemini(client, video_b_path)

    if is_first:
        eval_prompt = f"""You are a very critical and mindful expert video evaluator tasked with comparing two animated videos, A and B, to determine which more accurately and effectively addresses the User Prompt:

User Prompt:
{user_prompt}

**Important Constraints**:
- The winning scene must better adhere to the User Prompt.
- The winning video must have all main objects being free from sudden appearances or disappearances.
- The winning video must be free from too many unnecessary scene transitions.
- The winning video must have characters' movements free from unnaturally fast or slow speeds.

For each criterion below, assign a score of 1 (A wins), 0 (B wins), or 0.5 (TIE). Justify your score with 150-200 words per criterion.
- Visual Realism (Weight: 0.35): Which video has fewer non-realistic elements?
- Physical Commonsense (Weight: 0.3): Which video's actions are more internally logical?
- Video-Prompt Alignment (Weight: 0.35): Which video more accurately matches the User Prompt?

Steps:
1. Evaluate each criterion, assign score, provide 150-200 word justification.
2. Compute weighted scores. Apply -0.5 penalty for guideline violations.
3. If |s_A - s_B| < 0.05 → COMPARABLE. Otherwise A_BETTER or B_BETTER.

Output strictly as JSON:
```json
{{
"Reasoning_VisualRealism": "<150-200 words>",
"Reasoning_PhysicalCommonsense": "<150-200 words>",
"Reasoning_VideoPromptAlignment": "<150-200 words>",
"Decision": "<A_BETTER | B_BETTER | COMPARABLE>",
"WeightedScoreA": <float>,
"WeightedScoreB": <float>
}}
```"""
        contents = ["Video A:", file_a, "Video B:", file_b, eval_prompt]
    else:
        file_prev = upload_to_gemini(client, prev_video_path)
        eval_prompt = f"""You are a very critical and mindful expert video evaluator comparing two animated videos, A and B, for accuracy, coherence with a reference video, and adherence to the User Prompt:

User Prompt:
{user_prompt}

**Important Constraints**:
- The winning scene must better adhere to the User Prompt.
- The winning video must be coherent to the reference previous video.
- The winning video must be free from sudden object appearances/disappearances.
- The winning video must have characters' movements free from unnaturally fast or slow speeds.

Criteria:
- Visual Realism (Weight: 0.2)
- Physical Commonsense (Weight: 0.2)
- Video-Prompt Alignment (Weight: 0.2)
- Coherency with reference video (Weight: 0.4): character appearance and style consistency.

Steps:
1. Evaluate each criterion, assign score (1/0/0.5), 150-200 word justification.
2. Compute weighted scores. Apply -0.5 penalty for violations.
3. If |s_A - s_B| < 0.05 → COMPARABLE. Otherwise A_BETTER or B_BETTER.

Output strictly as JSON:
```json
{{
"Reasoning_VisualRealism": "<150-200 words>",
"Reasoning_PhysicalCommonsense": "<150-200 words>",
"Reasoning_VideoPromptAlignment": "<150-200 words>",
"Reasoning_Coherency": "<150-200 words>",
"Decision": "<A_BETTER | B_BETTER | COMPARABLE>",
"WeightedScoreA": <float>,
"WeightedScoreB": <float>
}}
```"""
        contents = ["Reference Previous Video:", file_prev, "Video A:", file_a, "Video B:", file_b, eval_prompt]

    print("    Requesting Gemini evaluation...")
    resp = client.models.generate_content(model="gemini-2.5-pro-preview-05-06", contents=contents)

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
# ARG PARSE & PATHS
# ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--target", required=True, help="Target folder name, e.g. lie_in_april1")
    parser.add_argument("--panel-count", type=int, default=PANEL_COUNT)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--manga-page", default="manga_page.jpg")
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
    target_dir            = PROJECT_ROOT / "target" / args.target
    manga_panels_dir      = target_dir / "manga_panels"
    storyboard_panels_dir = target_dir / "storyboard_panels"
    results_target_dir    = PROJECT_ROOT / "results" / args.target
    latest_iter_dir       = find_latest_iter_dir(results_target_dir)
    input_json_path       = latest_iter_dir / "bestScript.json"
    base_output_dir       = PROJECT_ROOT / "video-seedance" / "pipeline_generations" / args.target
    manga_page_path       = target_dir / args.manga_page

    for path, label in [
        (target_dir, "target"),
        (manga_panels_dir, "manga_panels"),
        (storyboard_panels_dir, "storyboard_panels"),
        (input_json_path, "bestScript.json"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not manga_page_path.exists():
        print(f"Warning: manga page not found: {manga_page_path}")

    return {
        "target_name":           args.target,
        "manga_panels_dir":      manga_panels_dir,
        "storyboard_panels_dir": storyboard_panels_dir,
        "input_json_path":       input_json_path,
        "base_output_dir":       base_output_dir,
        "manga_page_path":       manga_page_path,
    }

# ======================
# MAIN PIPELINE
# ======================
if __name__ == "__main__":
    args  = parse_args()
    paths = build_paths(args)

    target_name           = paths["target_name"]
    manga_panels_dir      = paths["manga_panels_dir"]
    storyboard_panels_dir = paths["storyboard_panels_dir"]
    input_json_path       = paths["input_json_path"]
    base_output_dir       = paths["base_output_dir"]
    manga_page_path       = paths["manga_page_path"]

    print("start:", datetime.now())
    print(f"target: {target_name}")
    print(f"script: {input_json_path}")
    print(f"output: {base_output_dir}")


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

    # gemini_client = genai.Client(api_key=GEMINI_API_KEY)  # used by Gemini comparison

    best_videos = []
    # prev_best_video_path = None  # used by Gemini comparison

    for i in range(1, args.panel_count + 1):
        print(f"\n======================================")
        print(f"Panel {i}")
        print(f"======================================")

        panel_dir = os.path.join(run_dir, f"panel_{i}")
        os.makedirs(panel_dir, exist_ok=True)

        final_panel_video_path = os.path.join(panel_dir, f"best_panel_{i}.mp4")

        if os.path.exists(final_panel_video_path):
            print(f"    [Resume] best_panel_{i}.mp4 already exists, skipping.")
            best_videos.append(final_panel_video_path)
            prev_best_video_path = final_panel_video_path
            continue

        local_img_path = storyboard_panels_dir / f"panel{i}.png"
        manga_img_path = manga_panels_dir / f"panel{i}.png"

        if not local_img_path.exists():
            print(f"    storyboard panel{i}.png not found, skipping.")
            continue
        if not manga_img_path.exists():
            print(f"    manga panel{i}.png not found, skipping.")
            continue

        shutil.copy(str(local_img_path), os.path.join(panel_dir, "input.png"))

        panel_data       = get_panel_data(full_json, i)
        optimal_duration = calculate_optimal_duration_int(str(manga_img_path), str(manga_page_path), panel_data["clip"])
        prompt           = PROMPT_FILE.read_text(encoding="utf-8").replace("<script>", json.dumps(panel_data, ensure_ascii=False))
        save_file(panel_data, panel_dir, "filtered_script.json", True)

        video_path      = os.path.join(panel_dir, "candidate_1.mp4")
        tasks_info_path = os.path.join(panel_dir, "tasks_info.json")

        if os.path.exists(video_path):
            print("    [Resume] candidate already exists, skipping submission.")
        else:
            if os.path.exists(tasks_info_path):
                print("    [Resume] found tasks_info.json, resuming poll...")
                with open(tasks_info_path, "r", encoding="utf-8") as f:
                    tasks_info = json.load(f)
                task_id = tasks_info["task_A"]
            else:
                print("    Uploading image to ImgBB...")
                image_url = upload_image_to_imgbb(str(local_img_path))

                print("    Submitting task to Seedance...")
                task_id = submit_seedance(prompt, image_url, duration=optimal_duration)
                save_file({"task_A": task_id}, panel_dir, "tasks_info.json", True)
                print(f"    Submitted: {task_id}")

            print("    Waiting for video...")
            url = poll_seedance(task_id)
            save_file(requests.get(url).content, panel_dir, "candidate_1.mp4")
            print("    Downloaded.")

        shutil.copy(video_path, final_panel_video_path)
        best_videos.append(final_panel_video_path)
        # prev_best_video_path = final_panel_video_path  # used by Gemini comparison

        # --- Gemini comparison (commented out) ---
        # video_2_path = os.path.join(panel_dir, "candidate_2.mp4")
        # task_b_id = submit_seedance(prompt, image_url, duration=optimal_duration)
        # url_b = poll_seedance(task_b_id)
        # save_file(requests.get(url_b).content, panel_dir, "candidate_2.mp4")
        # eval_vid_a = os.path.join(panel_dir, "video_A.mp4")
        # eval_vid_b = os.path.join(panel_dir, "video_B.mp4")
        # shutil.copy(video_path, eval_vid_a)
        # shutil.copy(video_2_path, eval_vid_b)
        # eval_text, eval_prompt_used = evaluate_videos(gemini_client, prompt, eval_vid_a, eval_vid_b, prev_best_video_path)
        # save_file(eval_prompt_used, panel_dir, "gemini_eval_prompt.txt")
        # save_file(eval_text, panel_dir, "gemini_eval_response.txt")
        # eval_result = extract_json_from_response(eval_text)
        # decision = eval_result.get("Decision", "A_BETTER")
        # best_src = video_2_path if decision == "B_BETTER" else video_path
        # shutil.copy(best_src, final_panel_video_path)
        # if os.path.exists(eval_vid_a): os.remove(eval_vid_a)
        # if os.path.exists(eval_vid_b): os.remove(eval_vid_b)

    print("\n======================================")
    print("Concatenating all best clips...")
    list_file_path = os.path.join(run_dir, "concat_list.txt")
    with open(list_file_path, "w", encoding="utf-8") as f:
        for vid in best_videos:
            f.write(f"file '{os.path.abspath(vid)}'\n")

    final_output_path = os.path.join(run_dir, "final_story.mp4")
    ffmpeg_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", final_output_path]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Done! Final video: {final_output_path}")
    except Exception as e:
        print(f"ffmpeg concat failed: {e}")

    print("DONE ->", run_dir)
