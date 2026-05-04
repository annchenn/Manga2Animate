import os
import sys
import json
import time
import shutil
import base64
import requests
import re
import subprocess
from datetime import datetime
import dashscope
from google import genai 
from PIL import Image 
import argparse
from pathlib import Path

# ======================
# SYSTEM PROMPTS
# ======================
WAN_SYSTEM_PROMPT = """1. Interpret the 'Script' JSON to generate a consistent video.
2. Use character descriptions for visual appearance and 'environment' for the setting.
3. Execute the actions described in 'clips' with natural motion and cinematic 1k quality.
4. Maintain visual continuity across the entire duration.
5. Avoid abrupt cuts; use smooth camera movements and transitions.
6. Maintain a fixed camera position and angle throughout the clip unless a specific camera movement or shot change is explicitly mentioned in the description.
7. No dialogue boxes or speech bubbles in any storyboard panels. Sound effects (onomatopoeia) are allowed, but no captions, subtitles, or narrative text.
8. Don't add any elements that are not described in the 'Script' JSON. Strictly adhere to the provided character descriptions and environment settings without introducing new objects, characters, or backgrounds that are not mentioned in the script.
9. Ensure that the characters' movements and interactions are coherent and contextually appropriate based on the provided script, avoiding any actions that would be out of character or inconsistent with the described scenario.
"""

# ======================
# CONFIG
# ======================
DASHSCOPE_API_KEY = ""
GEMINI_API_KEY = ""
IMGBB_API_KEY = ""

MODEL_NAME = "wan2.7-i2v"
VIDEO_DURATION = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Manga2Animate/
PANEL_COUNT = 6 # 設定處理的 panel 數量

dashscope.api_key = DASHSCOPE_API_KEY

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
    url = "https://api.imgbb.com/1/upload"
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    payload = {
        "key": IMGBB_API_KEY,
        "image": base64.b64encode(img_bytes)
    }
    res = requests.post(url, data=payload)
    data = res.json()

    if not data.get("success"):
        raise Exception(f"Upload failed: {data}")

    return data["data"]["url"]

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
    
    return {
        "characters": filtered_chars,
        "clip": clip
    }

# ======================
# ⭐ 新增：DYNAMIC DURATION CALCULATION (計算動態整數秒數)
# ======================
def calculate_optimal_duration_int(image_path, clip_data):
    base_time = 2.0 
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            area = width * height
            standard_area = 62500.0 # 預設 250x250 基準
            area_ratio = area / standard_area
            if(area_ratio < 1.0):
                visual_bonus = 0  # 小於基準圖，適度減少加成
            else:   
                visual_bonus = min(3.0, area_ratio * 1.0)
    except Exception:
        visual_bonus = 0.5

    description = clip_data.get("description", "")
    dialogues = re.findall(r"Dialogue\([^)]+\):\s*'([^']+)'", description)
    total_words = sum(len(d.split()) for d in dialogues)
    text_bonus = total_words / 10.0

    total_duration = base_time + visual_bonus + text_bonus
    # 限制影片長度在 2~5 秒之間，並四捨五入取整數給 WAN API
    optimal_duration_int = max(2, min(5, int(round(total_duration))))
    
    print(f"        ⏱️ [時長計算] 基礎:{base_time}s + 視覺:{visual_bonus:.2f}s + 對話({total_words}字):{text_bonus:.2f}s -> 動態指派: {optimal_duration_int} 秒")
    return optimal_duration_int

# ======================
# WAN API
# ======================
def submit_wan(prompt, image_url, duration=3):
    url = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": MODEL_NAME,
        "input": {
            "prompt": prompt,
            "media": [
                {"type": "first_frame", "url": image_url}
            ]
        },
        "parameters": {
            "size": "1K",
            "duration": duration,
            "prompt_extend": True,
            "watermark": False
        }
    }
    res = requests.post(url, headers=headers, json=payload)
    return res.json()

def poll_task(task_id):
    url = f"https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    
    while True:
        res = requests.get(url, headers=headers).json()
        status = res["output"]["task_status"]
        print(f"        Task {task_id[-5:]}... status: {status}")
        if status == "SUCCEEDED":
            return res["output"]["video_url"]
        if status in ["FAILED", "CANCELED"]:
            raise Exception(f"Task {task_id} failed: {res}")
        time.sleep(15)

# ======================
# GEMINI EVALUATION
# ======================
def upload_to_gemini(client, file_path):
    print(f"    上傳影片至 Gemini: {os.path.basename(file_path)}...")
    gemini_file = client.files.upload(file=file_path)
    
    while gemini_file.state.name == "PROCESSING":
        print("    處理中，等待 5 秒...")
        time.sleep(5)
        gemini_file = client.files.get(name=gemini_file.name)
        
    if gemini_file.state.name == "FAILED":
        raise Exception(f"Gemini File processing failed for {file_path}")
        
    return gemini_file

def evaluate_videos(client, user_prompt, video_a_path, video_b_path, prev_video_path=None):
    is_first = (prev_video_path is None)
    file_a = upload_to_gemini(client, video_a_path)
    file_b = upload_to_gemini(client, video_b_path)
    contents = []
    
    if is_first:
        eval_prompt = f"""You are a very critical and mindful expert video evaluator tasked with comparing two animated videos, A and B, to determine which more accurately and effectively addresses the User Prompt:

User Prompt:
{user_prompt}

Your task is to mindfully and thoroughly compare the two videos, reasoning step-by-step.
**Important Constraints**:
- The winning scene must better adhere to the User Prompt.
- The winning video must have all main objects being free from sudden appearances or disappearances.
- The winning video must be free from too many unnecessary scene transitions.
- The winning video must have characters’ movements free from unnaturally fast or slow speeds.

For each criterion below, assign a score of 1 (A wins), 0 (B wins), or 0.5 (TIE). Justify your score with 150-200 words per criterion.
- Visual Realism (Weight: 0.35): Which video has fewer non-realistic elements (e.g., distorted faces, impossible physics, sudden object appearances or disappearances, AI artifacts except watermark)? (If both are equally realistic and well-presented, mark TIE.)
- Physical Commonsense (Weight: 0.3): Which video’s character actions, environmental setting, events, movements, and dialogue (if any) are more internally logical and plausible given the scene description? (If both are equally logical, mark TIE.)
- Video-Prompt Alignment (Weight: 0.35): Which video more accurately matches the User Prompt in terms of visuals, actions, and constraints?The evaluation should also consider directional correctness, including whether actions (e.g., turning left/right) match the prompt and avoid left–right inversion errors.(If both match equally, mark TIE.)

Perform the Following Steps One-by-One:
1. Criterion Evaluation:
    - For each criterion, evaluate A vs. B based on the sub-criteria.
    - For each criterion, assign a score: 1 (A wins), 0 (B wins), 0.5 (TIE).
    - For each criterion, provide 150 to 200 words explanation, citing specific evidence
2. Weighted Score Calculation:
    - Apply guideline penalties: subtract 0.5 from s_A or s_B if violations were found.
    - Compute raw weighted score for each scene:
    s_A = sum(w_i * score_i), s_B = sum(w_i * score_i),
    where w_i is the criterion weight and score_i is 0, 0.5, or 1.
3. Final Decision:
    - If the absolute difference |s_A - s_B| is less than 0.05, output COMPARABLE.
    - Otherwise, output A_BETTER if s_A > s_B, or B_BETTER if s_B > s_A.
4. Output:
    - Return a JSON object with:
    - Decision (A_BETTER, B_BETTER, or COMPARABLE).
    - Final averaged weighted scores for Scene A and Scene B.
    ### Note: Be fair in your judgements. Avoid bias towards any scene. Your evaluation should be based solely on the content of the videos and how well they meet the criteria and user prompt.

Output strictly as JSON block:
```json
{{
"Reasoning_VisualRealism": "<150-200 words explanation here>",
"Reasoning_PhysicalCommonsense": "<150-200 words explanation here>",
"Reasoning_VideoPromptAlignment": "<150-200 words explanation here>",
"Decision": "<A_BETTER | B_BETTER | COMPARABLE>",
"WeightedScoreA": <float>,
"WeightedScoreB": <float>
}}
```
"""
        contents = ["Video A:", file_a, "Video B:", file_b, eval_prompt]
    else:
        file_prev = upload_to_gemini(client, prev_video_path)
        eval_prompt = f"""You are a very critical and mindful expert video evaluator tasked with comparing two animated videos, A and B, to determine which more accurately, effectively addresses the User Prompt, and be continuous and smooth from the first provided video:

User Prompt:
{user_prompt}

Your task is to mindfully and thoroughly compare the two videos, reasoning step-by-step.
**Important Constraints**:
- The winning scene must better adhere to the User Prompt.
- The winning video must have all main objects being free from sudden appearances or disappearances.
- The winning video must be coherent to the first provided animation video.
- The winning video must be free from too many unnecessary scene transitions.
- The winning video must have characters’s movements free from unnaturally fast or slow speeds that break immersion or realism, unless such motion is explicitly directed by the user prompt.

For each criterion below, assign a score of 1 (A wins), 0 (B wins), or 0.5 (TIE). Justify your score with 150-200 words per criterion. Your justification must reference observations from the scenes.
Avoid general, vague, or abstract reasoning. Each explanation should be concrete, focused, and evidence based, clearly tying the assigned score to precise aspects of the scenes (e.g.,dialogue flow, emotional clarity, pacing, visual cues, character motivation).

- Visual Realism (Weight: 0.2): Which video has fewer non-realistic elements (e.g., distorted faces, impossible physics, sudden object appearances or disappearances, AI artifacts except watermark)? (If both are equally realistic and well-presented, mark TIE.)
- Physical Commonsense (Weight: 0.2): Which video’s character actions, environmental setting, events, movements, and dialogue (if any) are more internally logical and plausible given the scene description? (If both are equally logical, mark TIE.)
- Video-Prompt Alignment (Weight: 0.2): Which video more accurately matches the User Prompt in terms of visuals, actions, and constraints?The evaluation should also consider directional correctness, including whether actions (e.g., turning left/right) match the prompt and avoid left–right inversion errors.(If both match equally, mark TIE.)
- Coherency (Weight: 0.4): Which video is more coherent to the first provided video in terms of character’s appearance and style?(If both match equally, mark TIE.)

Perform the Following Steps One-by-One:
1. Criterion Evaluation:
    - For each criterion, evaluate A vs. B based on the sub-criteria.
    - For each criterion, assign a score: 1 (A wins), 0 (B wins), 0.5 (TIE).
    - For each criterion, provide 150 to 200 words explanation, citing specific evidence
2. Weighted Score Calculation:
    - Apply guideline penalties: subtract 0.5 from s_A or s_B if violations were found.
    - Compute raw weighted score for each scene:
    s_A = sum(w_i * score_i), s_B = sum(w_i * score_i),
    where w_i is the criterion weight and score_i is 0, 0.5, or 1.
3. Final Decision:
    - If the absolute difference |s_A - s_B| is less than 0.05, output COMPARABLE.
    - Otherwise, output A_BETTER if s_A > s_B, or B_BETTER if s_B > s_A.
4. Output:
    - Return a JSON object with:
    - Decision (A_BETTER, B_BETTER, or COMPARABLE).
    - Final averaged weighted scores for Scene A and Scene B.
    ### Note: Be fair in your judgements. Avoid bias towards any scene. Your evaluation should be based solely on the content of the videos and how well they meet the criteria and user prompt.

Output strictly as JSON block:
```json
{{
"Reasoning_VisualRealism": "<150-200 words explanation here>",
"Reasoning_PhysicalCommonsense": "<150-200 words explanation here>",
"Reasoning_VideoPromptAlignment": "<150-200 words explanation here>",
"Reasoning_Coherency": "<150-200 words explanation here>",
"Decision": "<A_BETTER | B_BETTER | COMPARABLE>",
"WeightedScoreA": <float>,
"WeightedScoreB": <float>
}}
```
"""
        contents = ["Reference Previous Video:", file_prev, "Video A:", file_a, "Video B:", file_b, eval_prompt]

    print("    正在請求 Gemini 進行影片分析...")
    resp = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=contents
    )

    # 清理資源
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
        except:
            pass
    
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
            
    return json.loads(text)

# ======================
# MAIN PIPELINE
# ======================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--target",
        required=True,
        help="Target folder name, e.g. death_notebook, karakai, dangers_in_heart"
    )

    parser.add_argument(
        "--panel-count",
        type=int,
        default=PANEL_COUNT,
        help="Number of panels to process"
    )

    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from latest run folder"
    )

    parser.add_argument(
        "--storyboard-layout",
        default="2x3",
        help="Layout of bestStoryboard.png, default: 2x3"
    )

    return parser.parse_args()


def parse_storyboard_layout(layout_str):
    """
    Example:
    2x3 -> rows=2, cols=3
    3x2 -> rows=3, cols=2
    """
    try:
        rows, cols = layout_str.lower().split("x")
        return int(rows), int(cols)
    except Exception:
        raise ValueError(f"Invalid storyboard layout: {layout_str}, expected format like 2x3")


def find_latest_iter_dir(results_target_dir):
    """
    Find max iter folder under:
    Manga2Animate/results/{target}/iter0
    Manga2Animate/results/{target}/iter1
    ...
    """
    if not results_target_dir.exists():
        raise FileNotFoundError(f"Results target folder not found: {results_target_dir}")

    candidates = []

    for p in results_target_dir.iterdir():
        if not p.is_dir():
            continue

        match = re.fullmatch(r"iter(\d+)", p.name)
        if match:
            candidates.append((int(match.group(1)), p))

    if not candidates:
        raise FileNotFoundError(f"No iter folder found under: {results_target_dir}")

    latest_iter_dir = max(candidates, key=lambda x: x[0])[1]
    return latest_iter_dir


def prepare_storyboard_panels(best_storyboard_path, output_panel_dir, panel_count=6, layout="2x3"):
    """
    Split results/{target}/iter_max/bestStoryboard.png into:
    panel1.png, panel2.png, ..., panel6.png

    These are used as first-frame images for video generation.
    """
    output_panel_dir.mkdir(parents=True, exist_ok=True)

    expected_paths = [
        output_panel_dir / f"panel{i}.png"
        for i in range(1, panel_count + 1)
    ]

    if all(p.exists() for p in expected_paths):
        print(f"✅ Storyboard panels already exist: {output_panel_dir}")
        return output_panel_dir

    rows, cols = parse_storyboard_layout(layout)

    if rows * cols != panel_count:
        raise ValueError(
            f"storyboard layout {layout} gives {rows * cols} panels, "
            f"but panel_count is {panel_count}"
        )

    if not best_storyboard_path.exists():
        raise FileNotFoundError(f"bestStoryboard.png not found: {best_storyboard_path}")

    print(f"✂️ Splitting storyboard: {best_storyboard_path}")
    print(f"   layout = {rows}x{cols}, output = {output_panel_dir}")

    with Image.open(best_storyboard_path) as img:
        width, height = img.size

        panel_id = 1
        for r in range(rows):
            for c in range(cols):
                left = round(c * width / cols)
                upper = round(r * height / rows)
                right = round((c + 1) * width / cols)
                lower = round((r + 1) * height / rows)

                cropped = img.crop((left, upper, right, lower))
                out_path = output_panel_dir / f"panel{panel_id}.png"
                cropped.save(out_path)

                print(f"   saved panel{panel_id}: {out_path}")
                panel_id += 1

    return output_panel_dir


def build_paths(args):
    target_name = args.target

    target_dir = PROJECT_ROOT / "target" / target_name
    manga_panels_dir = target_dir / "manga_panels"
    storyboard_panels_dir = target_dir / "storyboard_panels"

    results_target_dir = PROJECT_ROOT / "results" / target_name
    latest_iter_dir = find_latest_iter_dir(results_target_dir)

    input_json_path = latest_iter_dir / "bestScript.json"

    base_output_dir = PROJECT_ROOT / "video" / "pipeline_generations" / target_name

    if not target_dir.exists():
        raise FileNotFoundError(f"Target folder not found: {target_dir}")

    if not manga_panels_dir.exists():
        raise FileNotFoundError(f"Manga panels folder not found: {manga_panels_dir}")

    if not storyboard_panels_dir.exists():
        raise FileNotFoundError(f"Storyboard panels folder not found: {storyboard_panels_dir}")

    if not input_json_path.exists():
        raise FileNotFoundError(f"bestScript.json not found: {input_json_path}")

    return {
        "target_name": target_name,
        "target_dir": target_dir,
        "manga_panels_dir": manga_panels_dir,
        "storyboard_panels_dir": storyboard_panels_dir,
        "results_target_dir": results_target_dir,
        "latest_iter_dir": latest_iter_dir,
        "input_json_path": input_json_path,
        "base_output_dir": base_output_dir,
    }

if __name__ == "__main__":
    args = parse_args()
    paths = build_paths(args)

    target_name = paths["target_name"]
    manga_panels_dir = paths["manga_panels_dir"]
    input_json_path = paths["input_json_path"]
    base_output_dir = paths["base_output_dir"]
    storyboard_panels_dir = paths["storyboard_panels_dir"]

    print("start:", datetime.now())
    print(f"📁 target: {target_name}")
    print(f"📄 script: {input_json_path}")
    print(f"🖼️ manga panels: {manga_panels_dir}")
    print(f"🎞️ storyboard panels: {storyboard_panels_dir}")
    print(f"📦 output: {base_output_dir}")

    os.makedirs(base_output_dir, exist_ok=True)

    run_dir = None
    if args.auto_resume:
        runs = sorted([d for d in os.listdir(base_output_dir) if d.startswith("run_")])
        if runs:
            run_dir = os.path.join(base_output_dir, runs[-1])
            print(f"🔄 [中斷回復] 偵測到過去的任務目錄，接續執行: {run_dir}")
            
    if not run_dir:
        run_dir = os.path.join(base_output_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"🆕 建立新任務目錄: {run_dir}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        full_json = json.load(f)
        
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    best_videos = []
    prev_best_video_path = None

    for i in range(1, args.panel_count + 1):
        print(f"\n======================================")
        print(f"🎬 處理 Panel {i}")
        print(f"======================================")
        
        panel_dir = os.path.join(run_dir, f"panel_{i}")
        os.makedirs(panel_dir, exist_ok=True)
        
        final_panel_video_path = os.path.join(panel_dir, f"best_panel_{i}.mp4")
        
        if os.path.exists(final_panel_video_path):
            print(f"    ⏭️ [中斷回復] 發現已完成的最終影片 best_panel_{i}.mp4，跳過此 Panel。")
            best_videos.append(final_panel_video_path)
            prev_best_video_path = final_panel_video_path
            continue

        local_img_path = storyboard_panels_dir / f"panel{i}.png"
        if not local_img_path.exists():
            print(f"❌ 找不到手切 storyboard panel 圖片 {local_img_path}，略過。")
            continue

        shutil.copy(str(local_img_path), os.path.join(panel_dir, "input.png"))
        
        panel_data = get_panel_data(full_json, i)
        manga_img_path = manga_panels_dir / f"panel{i}.png"
        if not manga_img_path.exists():
            print(f"❌ 找不到 manga panel 圖片 {manga_img_path}，略過。")
            continue

        optimal_duration = calculate_optimal_duration_int(str(manga_img_path), panel_data["clip"])
        wan_prompt = f"Rule:\n{WAN_SYSTEM_PROMPT.strip()}\n\nScript:\n{json.dumps(panel_data, ensure_ascii=False)}"
        save_file(wan_prompt, panel_dir, "wan_prompt.txt")
        save_file(panel_data, panel_dir, "filtered_script.json", True)
        
        # 使用 candidate 命名來存 Wan 的原始影片
        video_1_path = os.path.join(panel_dir, "candidate_1.mp4")
        video_2_path = os.path.join(panel_dir, "candidate_2.mp4")
        tasks_info_path = os.path.join(panel_dir, "tasks_info.json")
        
        if os.path.exists(video_1_path) and os.path.exists(video_2_path):
            print("    ⏭️ [中斷回復] 發現 candidate_1 與 candidate_2 已存在，跳過 Wan 影片生成。")
        else:
            if os.path.exists(tasks_info_path):
                print("    🔄 [中斷回復] 發現已提交的 tasks_info.json，載入 task_id 恢復 Polling...")
                with open(tasks_info_path, "r", encoding="utf-8") as f:
                    tasks_info = json.load(f)
                task_a_id = tasks_info["task_A"]["output"]["task_id"]
                task_b_id = tasks_info["task_B"]["output"]["task_id"]
            else:
                print("    上傳圖片至 ImgBB...")
                image_url = upload_image_to_imgbb(str(local_img_path))
                
                print("    🚀 提交任務 A 與 B 至 Wan...")
                task_a_res = submit_wan(wan_prompt, image_url, duration=optimal_duration)
                task_b_res = submit_wan(wan_prompt, image_url, duration=optimal_duration)
                
                task_a_id = task_a_res["output"]["task_id"]
                task_b_id = task_b_res["output"]["task_id"]
                
                save_file({"task_A": task_a_res, "task_B": task_b_res}, panel_dir, "tasks_info.json", True)
                print(f"    任務提交成功: Task A [{task_a_id}], Task B [{task_b_id}]")

            print("    等待影片生成...")
            if not os.path.exists(video_1_path):
                url_a = poll_task(task_a_id)
                save_file(requests.get(url_a).content, panel_dir, "candidate_1.mp4")
            if not os.path.exists(video_2_path):
                url_b = poll_task(task_b_id)
                save_file(requests.get(url_b).content, panel_dir, "candidate_2.mp4")
            print("    ✅ 雙軌影片下載完成。")
        
        # --- ⭐ 關鍵改動：評估前將候選影片複製並重新命名為 video_A / video_B ---
        eval_vid_a_path = os.path.join(panel_dir, "video_A.mp4")
        eval_vid_b_path = os.path.join(panel_dir, "video_B.mp4")
        shutil.copy(video_1_path, eval_vid_a_path)
        shutil.copy(video_2_path, eval_vid_b_path)

        print("    🤖 呼叫 Gemini 進行專業評測...")
        try:
            # 送入的是命名為 video_A 與 video_B 的檔案
            eval_text, eval_prompt_used = evaluate_videos(
                gemini_client, 
                wan_prompt, 
                eval_vid_a_path, 
                eval_vid_b_path, 
                prev_best_video_path
            )
            
            save_file(eval_prompt_used, panel_dir, "gemini_eval_prompt.txt")
            save_file(eval_text, panel_dir, "gemini_eval_response.txt")
            
            eval_result = extract_json_from_response(eval_text)
            decision = eval_result.get("Decision", "A_BETTER")
        except Exception as e:
            print(f"    ⚠️ 評估過程發生錯誤，預設選擇 A。錯誤: {e}")
            decision = "A_BETTER"
            
        print(f"    🏆 評估結果: {decision}")
        
        # 6. 選定最佳影片並記錄 (這裡從 candidate 複製過去，確保來源正確)
        best_video_for_panel = video_2_path if decision == "B_BETTER" else video_1_path
        shutil.copy(best_video_for_panel, final_panel_video_path)
        
        best_videos.append(final_panel_video_path)
        prev_best_video_path = final_panel_video_path

        # --- 清理暫存檔 ---
        if os.path.exists(eval_vid_a_path): os.remove(eval_vid_a_path)
        if os.path.exists(eval_vid_b_path): os.remove(eval_vid_b_path)
        
    print("\n======================================")
    print("🎥 開始串接所有最佳片段...")
    list_file_path = os.path.join(run_dir, "concat_list.txt")
    with open(list_file_path, "w", encoding="utf-8") as f:
        for vid in best_videos:
            f.write(f"file '{os.path.abspath(vid).replace('\\\\', '/')}'\n")
            
    final_output_path = os.path.join(run_dir, "final_story.mp4")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file_path, 
        "-c", "copy", final_output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"🎉 影片串接完成！最終成果: {final_output_path}")
    except Exception as e:
        print(f"❌ 影片串接失敗，請確認是否已安裝 ffmpeg。錯誤: {e}")

    print("DONE ->", run_dir)