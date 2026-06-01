#!/usr/bin/env python3
"""
Generate multiple complete full-length storyboard videos using Veo 3.1, then evaluate the completed full videos against each other.

Pipeline:
1. Generate full_candidate_1 from bestStoryboard.png as a Veo 3.1 reference image, then extend it step-by-step until the requested full length is reached.
2. Generate full_candidate_2 independently from the storyboard and extend it step-by-step until the same requested full length is reached.
3. Compare the final cumulative full videos with the Manga2Animate/Gemini A/B rubric, prioritizing within-video character identity consistency.
4. Save the selected complete video as finalFullVideo.mp4.

Important notes:
- Veo 3.1 reference-image generation requires duration_seconds=8.
- Veo 3.1 video extension works only for Veo-generated videos and expects 720p inputs.
- Extension output is cumulative: the returned video already contains the input video plus the new continuation.

Typical usage from the Manga2Animate project root:

    python video/pipeline_veo31_storyboard_extend_two_eval.py -t karakai

Useful overrides:

    python video/pipeline_veo31_storyboard_full_candidates_eval.py -t karakai --extend-steps 2
    python video/pipeline_veo31_storyboard_full_candidates_eval.py -t karakai --full-video-candidates 2

Authentication:
- Gemini API key mode: set GEMINI_API_KEY or GOOGLE_API_KEY.
- Vertex AI mode: set GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional at runtime
    load_dotenv = None

try:
    from google import genai
    from google.genai import types
except Exception:  # Delay the hard failure until main(), so --help still works when possible.
    genai = None
    types = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "prompts" / "video_generation_storyboard.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "video" / "pipeline_generations_veo31_storyboard_full_candidates_v2"

# Veo 3.1 reference-image generation requires an 8-second duration.
REFERENCE_IMAGE_DURATION_SECONDS = 8
# Veo 3.1 extension also uses durationSeconds=8 in the parameter table, while the
# generated continuation adds roughly 7 seconds and returns a cumulative video.
EXTENSION_DURATION_SECONDS = 8


# ======================
# Small utilities
# ======================
def save_file(content: Any, folder: Path, filename: str, is_json: bool = False) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif is_json:
        path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text(str(content), encoding="utf-8")
    return path


def model_to_jsonable(obj: Any) -> Any:
    """Best-effort conversion for google-genai pydantic-like objects."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [model_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): model_to_jsonable(v) for k, v in obj.items()}
    for method_name in ("model_dump", "to_json_dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return model_to_jsonable(method())
            except Exception:
                pass
    return str(obj)


def require_google_genai() -> None:
    if genai is None or types is None:
        raise RuntimeError(
            "google-genai is not installed. Install or upgrade it with: "
            "pip install --upgrade google-genai python-dotenv"
        )
    if not hasattr(types, "VideoGenerationReferenceImage"):
        raise RuntimeError(
            "Your google-genai version does not expose VideoGenerationReferenceImage. "
            "Please upgrade: pip install --upgrade google-genai"
        )


def truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def copy_if_local(src: str, dst: Path) -> str:
    src_path = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src_path.exists():
        shutil.copy2(src_path, dst)
        return str(dst)
    return src


# ======================
# Path helpers
# ======================
def find_latest_iter_dir(results_target_dir: Path) -> Path:
    if not results_target_dir.exists():
        raise FileNotFoundError(f"Results target folder not found: {results_target_dir}")

    candidates = []
    for p in results_target_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"iter(\d+)", p.name)
        if m:
            candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(f"No iter folder found under: {results_target_dir}")

    return max(candidates, key=lambda x: x[0])[1]


def resolve_project_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_paths(args: argparse.Namespace) -> Dict[str, Path]:
    results_target_dir = PROJECT_ROOT / "results" / args.target
    latest_iter_dir = find_latest_iter_dir(results_target_dir)

    script_path = resolve_project_path(args.script_json) or (latest_iter_dir / "bestScript.json")
    storyboard_path = resolve_project_path(args.storyboard_image) or (latest_iter_dir / "bestStoryboard.png")
    prompt_file = resolve_project_path(args.prompt_file) or DEFAULT_PROMPT_FILE
    output_root = resolve_project_path(args.output_root) or (DEFAULT_OUTPUT_ROOT / args.target)

    for path, label in [
        (script_path, "bestScript.json"),
        (storyboard_path, "bestStoryboard.png"),
        (prompt_file, "prompt file"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    return {
        "results_target_dir": results_target_dir,
        "latest_iter_dir": latest_iter_dir,
        "script_path": script_path,
        "storyboard_path": storyboard_path,
        "prompt_file": prompt_file,
        "output_root": output_root,
    }


# ======================
# Prompt construction
# ======================
def compact_script(full_json: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields that are useful for video generation."""
    compact: Dict[str, Any] = {}

    if "characters" in full_json:
        compact["characters"] = []
        for c in full_json.get("characters", []):
            item = {"id": c.get("id")}
            for key in ("name", "description", "appearance", "visual_description", "personality"):
                if c.get(key):
                    item[key] = c[key]
            compact["characters"].append(item)

    compact["clips"] = []
    for clip in full_json.get("clips", []):
        item = {"id": clip.get("id")}
        for key in ("environment", "description", "camera", "duration"):
            if clip.get(key):
                item[key] = clip[key]
        compact["clips"].append(item)

    return compact


def build_storyboard_prompt(full_json: Dict[str, Any], prompt_file: Path, use_compact_script: bool) -> str:
    script_for_prompt = compact_script(full_json) if use_compact_script else full_json
    script_text = json.dumps(script_for_prompt, ensure_ascii=False, separators=(",", ":"))

    template = prompt_file.read_text(encoding="utf-8")
    if "<script>" in template:
        prompt = template.replace("<script>", script_text)
    else:
        prompt = f"{template.strip()}\n\nScript:\n{script_text}"

    storyboard_specific_rules = """
Additional Veo 3.1 full-storyboard reference rules for a paced two-segment plan:
- Treat the reference image as the complete storyboard for the whole video, not as a single still frame to reproduce.
- Assume the storyboard panels are ordered left-to-right, top-to-bottom.
- For this INITIAL 8-second generation, animate ONLY the FIRST HALF of the storyboard: panels 1 to 3.
- Use this initial segment to establish the scene, characters, and the first narrative beat(s).
- Do NOT try to cover the entire storyboard. Do NOT rush to the ending. Do NOT squeeze panels 4 to 6 into this first segment.
- The first segment should feel like one continuous animated scene, not a slideshow of many panels.
- Preserve the storyboard's style, composition cues, action direction, and spatial continuity, but do not overfit to small drawing differences between panels.
- Character identity must be internally consistent within the same generated video: a recurring character may slightly drift in detail, lighting, angle, or stylization, but must remain recognizable as the same person throughout the clip.
- Do not let the same character become a different person mid-video. Avoid sudden changes in face shape, hairstyle, hair color, eye shape, outfit, body proportions, age, or gender unless the script explicitly says so.
- If the same character appears differently across storyboard panels, interpret those drawings as the same character under different poses/expressions, not as separate people.
- Do not show panel borders, gutters, captions, speech bubbles, dialogue text, or sound-effect text in the output video.
- When a subject is moving across panels, infer the in-between motion instead of abruptly teleporting or resetting the subject.
- It is preferred that this segment ends naturally before the whole story is complete, leaving the remaining beats for the extension step.
""".strip()

    return f"{prompt.strip()}\n\n{storyboard_specific_rules}\n"

def build_extension_prompt(full_json: Dict[str, Any], base_prompt: str, step_index: int, total_steps: int) -> str:
    script_for_prompt = compact_script(full_json)
    script_text = json.dumps(script_for_prompt, ensure_ascii=False, separators=(",", ":"))

    if step_index == 1:
        segment_instruction = """
This is the SECOND segment of a paced two-segment video plan.
Animate ONLY the SECOND HALF of the storyboard: panels 4 to 6.
The first segment already covered panels 1 to 3. Continue naturally from the last frame and cover the remaining later story beats.
Do not restart the story, do not recap the opening, and do not jump around between all panels. Focus on finishing the unresolved later half of the storyboard.
""".strip()
    else:
        segment_instruction = f"""
This is extra extension step {step_index}. The first two segments should already cover panels 1 to 6.
Only continue unresolved action if the story still needs a brief ending beat. Do not restart earlier scenes.
""".strip()

    return f"""
Continue the previously generated Veo storyboard video. This is extension step {step_index} of {total_steps}.

The input video already covers the opening part of the storyboard. Continue from the final frame naturally.

{segment_instruction}

Use the script and storyboard intent below to infer what should happen after the current ending:
{script_text}

Continuation requirements:
- Continue the exact same characters, art style, lighting, camera language, and spatial layout from the input video.
- Character identity must stay internally consistent across the cumulative video. Some visual drift is acceptable, but the same character must not turn into a different person after extension.
- Preserve each recurring character's recognizable identity: face structure, hairstyle, hair color, outfit, body proportions, age, and gender should remain stable unless the script explicitly requires a change.
- Maintain smooth physical continuity from the last frame of the input video.
- Avoid sudden character swaps, duplicate characters, abrupt scene resets, teleporting, or unexplained object disappearances.
- Do not render manga panel borders, gutters, speech bubbles, captions, subtitles, dialogue text, or sound-effect text.
- If the first segment stopped mid-action, complete that action first, then proceed to the next beat.
- Keep the pacing clear and cinematic. The continuation should feel like the second half of the same animation, not a separate clip.
- Do not rush through all remaining events in a fast montage. Let the later beats breathe and unfold as a continuous scene.
""".strip()



# ======================
# GenAI / Veo helpers
# ======================
def make_genai_client(args: argparse.Namespace):
    require_google_genai()

    if args.api_backend == "vertex":
        use_vertex = True
    elif args.api_backend == "gemini":
        use_vertex = False
    else:
        use_vertex = truthy_env("GOOGLE_GENAI_USE_VERTEXAI")

    if use_vertex:
        project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT")
        location = args.location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if not project:
            raise RuntimeError(
                "Vertex AI mode requires GOOGLE_CLOUD_PROJECT or --project. "
                "Also make sure Google Cloud authentication is configured."
            )
        print(f"🔐 Using Vertex AI backend: project={project}, location={location}")
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API mode requires GEMINI_API_KEY, GOOGLE_API_KEY, or --api-key.")
    print("🔐 Using Gemini API key backend")
    return genai.Client(api_key=api_key)


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime in {"image/png", "image/jpeg"}:
        return mime
    raise ValueError(f"Unsupported storyboard image MIME type: {mime}. Please use PNG or JPEG: {path}")


def load_reference_image(args: argparse.Namespace, storyboard_path: Path):
    require_google_genai()

    if args.storyboard_gcs_uri:
        mime = args.storyboard_mime_type or guess_mime_type(storyboard_path)
        return types.Image(gcs_uri=args.storyboard_gcs_uri, mime_type=mime)

    size_mb = storyboard_path.stat().st_size / 1024 / 1024
    if size_mb > args.max_image_mb:
        raise ValueError(
            f"Storyboard image is {size_mb:.2f} MB, larger than --max-image-mb={args.max_image_mb}. "
            "Compress or resize the storyboard, or upload it to GCS and pass --storyboard-gcs-uri."
        )

    return types.Image.from_file(location=str(storyboard_path))


def build_initial_generate_config(args: argparse.Namespace, reference_image) -> Any:
    require_google_genai()

    if args.duration_seconds != REFERENCE_IMAGE_DURATION_SECONDS:
        print(
            f"⚠️ Veo 3.1 reference-image generation requires "
            f"{REFERENCE_IMAGE_DURATION_SECONDS}s; overriding --duration-seconds={args.duration_seconds}."
        )

    ref = types.VideoGenerationReferenceImage(
        image=reference_image,
        reference_type="asset",
    )

    config_kwargs: Dict[str, Any] = {
        "reference_images": [ref],
        "number_of_videos": 1,
        "duration_seconds": REFERENCE_IMAGE_DURATION_SECONDS,
        "aspect_ratio": args.aspect_ratio,
        "resolution": args.resolution,
        "person_generation": args.person_generation,
    }

    if args.seed is not None:
        config_kwargs["seed"] = args.seed
    if args.output_gcs_uri:
        config_kwargs["output_gcs_uri"] = args.output_gcs_uri

    # Keep only fields accepted by Veo 3.1 reference-image mode.
    return types.GenerateVideosConfig(**config_kwargs)


def build_extension_config(args: argparse.Namespace) -> Any:
    require_google_genai()

    config_kwargs: Dict[str, Any] = {
        "number_of_videos": 1,
        "duration_seconds": EXTENSION_DURATION_SECONDS,
        "resolution": "720p",
    }
    if args.output_gcs_uri:
        config_kwargs["output_gcs_uri"] = args.output_gcs_uri

    # Keep extension config minimal. Do not pass reference_images, person_generation,
    # or extra SDK flags in extension mode.
    return types.GenerateVideosConfig(**config_kwargs)


def poll_operation(client, operation, interval_seconds: int, timeout_seconds: int, save_dir: Path):
    start = time.time()
    operation_name = getattr(operation, "name", None)
    save_file({"operation_name": operation_name}, save_dir, "operation_started.json", True)

    while not getattr(operation, "done", False):
        elapsed = int(time.time() - start)
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Veo operation timed out after {timeout_seconds}s: {operation_name}")
        print(f"⏳ Waiting for Veo operation... elapsed={elapsed}s name={operation_name}")
        time.sleep(interval_seconds)
        operation = client.operations.get(operation)
        save_file(model_to_jsonable(operation), save_dir, "operation_latest.json", True)

    save_file(model_to_jsonable(operation), save_dir, "operation_done.json", True)
    return operation


def save_generated_video(client, operation, output_path: Path) -> Dict[str, Any]:
    if not getattr(operation, "response", None):
        raise RuntimeError(f"Veo operation finished without response: {operation}")

    generated_videos = operation.response.generated_videos
    if not generated_videos:
        raise RuntimeError(f"Veo operation response contains no generated_videos: {operation.response}")

    generated_video = generated_videos[0]
    video_obj = generated_video.video

    # If Vertex writes directly to GCS, the SDK may return only a URI.
    uri = getattr(video_obj, "uri", None)

    # Gemini API usually returns a file handle that must be downloaded before saving.
    try:
        client.files.download(file=video_obj)
    except TypeError:
        try:
            client.files.download(video_obj)
        except Exception:
            pass
    except Exception:
        # Some Vertex/GCS flows do not need client.files.download().
        pass

    saved = False
    save_method = getattr(video_obj, "save", None)
    if callable(save_method):
        try:
            save_method(str(output_path))
            saved = output_path.exists()
        except Exception:
            saved = False

    if not saved:
        video_bytes = getattr(video_obj, "video_bytes", None)
        if video_bytes:
            output_path.write_bytes(video_bytes)
            saved = True

    if saved:
        path_or_uri = str(output_path)
    elif uri:
        path_or_uri = uri
    else:
        raise RuntimeError("Could not save generated video locally and no video URI was returned.")

    return {
        "path": path_or_uri,
        "video_obj": video_obj,
        "generated_video": generated_video,
        "operation": operation,
    }


def generate_initial_candidate(
    client,
    args: argparse.Namespace,
    prompt: str,
    reference_image,
    run_dir: Path,
    candidate_index: int,
) -> Dict[str, Any]:
    candidate_dir = run_dir / "initial" / f"candidate_{candidate_index}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    output_path = candidate_dir / f"veo31_storyboard_initial_candidate_{candidate_index}.mp4"

    config = build_initial_generate_config(args, reference_image)
    save_file(model_to_jsonable(config), candidate_dir, "veo_config.json", True)

    print(f"🚀 Submitting initial candidate_{candidate_index} to Veo 3.1: model={args.model}")
    operation = client.models.generate_videos(
        model=args.model,
        prompt=prompt,
        config=config,
    )

    operation = poll_operation(
        client=client,
        operation=operation,
        interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        save_dir=candidate_dir,
    )

    result = save_generated_video(client, operation, output_path)
    print(f"✅ initial candidate_{candidate_index} saved/result: {result['path']}")
    return result


def generate_extension_candidate(
    client,
    args: argparse.Namespace,
    prompt: str,
    input_video_obj,
    run_dir: Path,
    step_index: int,
    candidate_index: int,
) -> Dict[str, Any]:
    candidate_dir = run_dir / f"extend_step_{step_index}" / f"candidate_{candidate_index}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    output_path = candidate_dir / f"veo31_storyboard_extend_step_{step_index}_candidate_{candidate_index}.mp4"

    config = build_extension_config(args)
    save_file(model_to_jsonable(config), candidate_dir, "veo_extension_config.json", True)

    print(f"🚀 Submitting extension step {step_index} candidate_{candidate_index} to Veo 3.1: model={args.model}")
    operation = client.models.generate_videos(
        model=args.model,
        video=input_video_obj,
        prompt=prompt,
        config=config,
    )

    operation = poll_operation(
        client=client,
        operation=operation,
        interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        save_dir=candidate_dir,
    )

    result = save_generated_video(client, operation, output_path)
    print(f"✅ extension step {step_index} candidate_{candidate_index} saved/result: {result['path']}")
    return result


# ======================
# Gemini video evaluation / best-video selection
# ======================
def upload_to_gemini_for_eval(client, file_path: str):
    """Upload a local video file to Gemini and wait until it is ready for video understanding."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file for evaluation does not exist locally: {file_path}")

    print(f"    上傳影片至 Gemini evaluator: {path.name}...")
    try:
        gemini_file = client.files.upload(file=str(path))
    except TypeError:
        gemini_file = client.files.upload(file_path=str(path))

    while getattr(gemini_file, "state", None) and gemini_file.state.name == "PROCESSING":
        print("    Gemini file processing; waiting 5 seconds...")
        time.sleep(5)
        gemini_file = client.files.get(name=gemini_file.name)

    if getattr(gemini_file, "state", None) and gemini_file.state.name == "FAILED":
        raise RuntimeError(f"Gemini File processing failed for {file_path}")

    return gemini_file


def evaluate_videos_original_rules(
    client,
    user_prompt: str,
    video_a_path: str,
    video_b_path: str,
    eval_model: str,
    prev_video_path: Optional[str] = None,
):
    """
    Compare two videos using a Manga2Animate-style A/B rubric, but with an added
    within-video character identity consistency priority.

    Important distinction:
    - The video does NOT need to perfectly match the exact face in the storyboard.
    - It DOES need to keep each recurring character internally consistent within the same video.
    """
    is_first = prev_video_path is None
    file_a = file_b = file_prev = None
    try:
        file_a = upload_to_gemini_for_eval(client, video_a_path)
        file_b = upload_to_gemini_for_eval(client, video_b_path)

        if is_first:
            eval_prompt = f"""You are a very critical and mindful expert video evaluator tasked with comparing two animated videos, A and B, to determine which one is the better generated animation for the User Prompt.

User Prompt:
{user_prompt}

Core evaluation priority:
The most important issue is WITHIN-VIDEO CHARACTER IDENTITY CONSISTENCY. A character is allowed to drift slightly from the storyboard reference or from exact manga details. However, within the same generated video, each recurring character must stay recognizable as the same person from beginning to end. Penalize a video heavily if the same character suddenly changes into a different-looking person mid-video, swaps identity with another character, changes hairstyle/hair color/face shape/outfit/body proportions/age/gender without story justification, or appears as inconsistent duplicate versions of themselves.

Do NOT penalize small changes caused by lighting, camera angle, facial expression, motion blur, stylization, or normal animation deformation as long as the viewer can still clearly perceive the same character identity.

**Important Constraints**:
- The winning video must keep every recurring character internally consistent within the video.
- The winning video must better adhere to the User Prompt.
- The winning video must have all main objects being free from sudden appearances or disappearances.
- The winning video must be free from too many unnecessary scene transitions.
- The winning video must have characters' movements free from unnaturally fast or slow speeds.

For each criterion below, assign a score of 1 (A wins), 0 (B wins), or 0.5 (TIE). Justify your score with concrete video evidence.
- Character Identity Consistency (Weight: 0.40): Which video better keeps the same recurring character(s) recognizable as the same person throughout the whole clip? Focus on internal consistency inside the generated video, not exact matching to the original storyboard drawing. Penalize sudden identity changes, character swaps, unstable face/hair/outfit/body, or duplicate inconsistent versions. (If both are equally consistent, mark TIE.)
- Video-Prompt Alignment (Weight: 0.25): Which video more accurately matches the User Prompt in terms of visuals, actions, story beats, constraints, and action direction? (If both match equally, mark TIE.)
- Visual Realism / Animation Quality (Weight: 0.20): Which video has fewer distracting AI artifacts, distorted faces/bodies, flickering, broken objects, or impossible visual changes? Minor style changes are acceptable if identity remains consistent. (If both are equally good, mark TIE.)
- Physical Commonsense (Weight: 0.15): Which video's actions, environmental setting, events, movements, and object continuity are more internally logical and plausible? (If both are equally logical, mark TIE.)

Perform the Following Steps One-by-One:
1. Criterion Evaluation:
    - For each criterion, evaluate A vs. B based on the sub-criteria.
    - For each criterion, assign a score: 1 (A wins), 0 (B wins), 0.5 (TIE).
    - For each criterion, provide 100 to 180 words explanation, citing specific evidence from the videos.
2. Weighted Score Calculation:
    - Apply guideline penalties: subtract 0.5 from the relevant video score if there is a severe identity swap, major character replacement, or sudden unexplained identity change.
    - Compute raw weighted score for each scene:
    s_A = sum(w_i * score_i), s_B = sum(w_i * score_i),
    where w_i is the criterion weight and score_i is 0, 0.5, or 1.
3. Final Decision:
    - If one video has a clear severe within-video identity inconsistency and the other does not, prefer the consistent video unless it is otherwise unusable.
    - If the absolute difference |s_A - s_B| is less than 0.05, output COMPARABLE.
    - Otherwise, output A_BETTER if s_A > s_B, or B_BETTER if s_B > s_A.
4. Output:
    - Return a JSON object with Decision and weighted scores.

Output strictly as JSON block:
```json
{{
"Reasoning_CharacterIdentityConsistency": "<100-180 words explanation here>",
"Reasoning_VideoPromptAlignment": "<100-180 words explanation here>",
"Reasoning_VisualRealism": "<100-180 words explanation here>",
"Reasoning_PhysicalCommonsense": "<100-180 words explanation here>",
"Decision": "<A_BETTER | B_BETTER | COMPARABLE>",
"WeightedScoreA": <float>,
"WeightedScoreB": <float>
}}
```
"""
            contents = ["Video A:", file_a, "Video B:", file_b, eval_prompt]
        else:
            file_prev = upload_to_gemini_for_eval(client, prev_video_path)
            eval_prompt = f"""You are a very critical and mindful expert video evaluator tasked with comparing two extended animated videos, A and B, to determine which one is the better continuation of the Reference Previous Video and the User Prompt.

User Prompt:
{user_prompt}

Core evaluation priority:
The most important issue is WITHIN-VIDEO CHARACTER IDENTITY CONSISTENCY across the cumulative video. A character may drift slightly from the storyboard or original manga details, but within each final video the same recurring character must remain recognizable as the same person across both the previous part and the newly extended part. Penalize a candidate heavily if the extension turns the character into a different person, swaps identities between characters, changes hairstyle/hair color/face shape/outfit/body proportions/age/gender without story justification, or creates inconsistent duplicate versions of the same character.

Do NOT penalize small changes caused by lighting, camera angle, facial expression, motion blur, stylization, or normal animation deformation as long as the viewer can still clearly perceive the same character identity.

**Important Constraints**:
- The winning video must preserve each recurring character's identity across the cumulative video and the newly extended portion.
- The winning video must be coherent to the Reference Previous Video.
- The winning video must better adhere to the User Prompt and continue the unresolved story beats rather than restart the scene.
- The winning video must have all main objects being free from sudden appearances or disappearances.
- The winning video must be free from too many unnecessary scene transitions.
- The winning video must have characters' movements free from unnaturally fast or slow speeds that break immersion or realism.

For each criterion below, assign a score of 1 (A wins), 0 (B wins), or 0.5 (TIE). Justify your score with concrete video evidence.
- Character Identity Consistency (Weight: 0.35): Which extended video better keeps every recurring character recognizable as the same person throughout the cumulative output, especially after the extension begins? Focus on internal identity consistency, not exact matching to the storyboard. (If both are equally consistent, mark TIE.)
- Continuation Coherency (Weight: 0.30): Which video more naturally continues from the Reference Previous Video in character appearance, style, motion, camera continuity, scene continuity, and story progression? (If both continue equally well, mark TIE.)
- Video-Prompt Alignment (Weight: 0.20): Which video more accurately completes the later/unresolved story beats from the User Prompt without restarting or looping? (If both match equally, mark TIE.)
- Visual Realism / Animation Quality (Weight: 0.10): Which video has fewer distracting artifacts, distorted faces/bodies, flickering, or broken visual transitions? (If both are equally good, mark TIE.)
- Physical Commonsense (Weight: 0.05): Which video's character actions, object interactions, environmental behavior, and motion continuity are more physically plausible? (If both are equally logical, mark TIE.)

Special note for Veo extension outputs: Video A and Video B may each be cumulative videos that include the previous input portion plus a newly generated continuation. Evaluate which one has the better continuation after the point corresponding to the Reference Previous Video, while still checking the whole cumulative output for identity inconsistencies and artifacts.

Perform the Following Steps One-by-One:
1. Criterion Evaluation:
    - For each criterion, evaluate A vs. B based on the sub-criteria.
    - For each criterion, assign a score: 1 (A wins), 0 (B wins), 0.5 (TIE).
    - For each criterion, provide 100 to 180 words explanation, citing specific evidence from the videos.
2. Weighted Score Calculation:
    - Apply guideline penalties: subtract 0.5 from the relevant video score if there is a severe identity swap, major character replacement, or sudden unexplained identity change.
    - Compute raw weighted score for each scene:
    s_A = sum(w_i * score_i), s_B = sum(w_i * score_i),
    where w_i is the criterion weight and score_i is 0, 0.5, or 1.
3. Final Decision:
    - If one video has a clear severe identity inconsistency and the other does not, prefer the consistent video unless it is otherwise unusable.
    - If the absolute difference |s_A - s_B| is less than 0.05, output COMPARABLE.
    - Otherwise, output A_BETTER if s_A > s_B, or B_BETTER if s_B > s_A.
4. Output:
    - Return a JSON object with Decision and weighted scores.

Output strictly as JSON block:
```json
{{
"Reasoning_CharacterIdentityConsistency": "<100-180 words explanation here>",
"Reasoning_ContinuationCoherency": "<100-180 words explanation here>",
"Reasoning_VideoPromptAlignment": "<100-180 words explanation here>",
"Reasoning_VisualRealism": "<100-180 words explanation here>",
"Reasoning_PhysicalCommonsense": "<100-180 words explanation here>",
"Decision": "<A_BETTER | B_BETTER | COMPARABLE>",
"WeightedScoreA": <float>,
"WeightedScoreB": <float>
}}
```
"""
            contents = ["Reference Previous Video:", file_prev, "Video A:", file_a, "Video B:", file_b, eval_prompt]

        print(f"    🤖 呼叫 Gemini evaluator: {eval_model}")
        resp = client.models.generate_content(model=eval_model, contents=contents)
        return resp.text.strip(), eval_prompt
    finally:
        for f in (file_a, file_b, file_prev):
            if f is None:
                continue
            try:
                client.files.delete(name=f.name)
            except Exception as e:
                print(f"    ⚠️ Could not delete temporary Gemini file {getattr(f, 'name', '<unknown>')}: {e}")

def extract_json_from_response(text: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    return json.loads(text)


def choose_best_video(
    client,
    args: argparse.Namespace,
    prompt: str,
    candidate_results: list[Dict[str, Any]],
    selection_dir: Path,
    best_filename: str,
    selection_filename: str,
    prev_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    candidate_paths = [str(r["path"]) for r in candidate_results]
    selection_dir.mkdir(parents=True, exist_ok=True)

    if len(candidate_paths) < 2:
        only = candidate_paths[0]
        best_path = selection_dir / best_filename
        best_video_output = copy_if_local(only, best_path)
        result = {
            "Decision": "ONLY_ONE_CANDIDATE",
            "best_candidate": 1,
            "best_video_path": best_video_output,
            "candidate_paths": candidate_paths,
        }
        save_file(result, selection_dir, selection_filename, True)
        return result

    if len(candidate_paths) > 2:
        print("⚠️ 目前會用 identity-consistency-aware A/B 規則，只比較前兩個 candidates。")

    video_a_path, video_b_path = candidate_paths[0], candidate_paths[1]

    eval_text, eval_prompt_used = evaluate_videos_original_rules(
        client=client,
        user_prompt=prompt,
        video_a_path=video_a_path,
        video_b_path=video_b_path,
        eval_model=args.eval_model,
        prev_video_path=prev_video_path,
    )
    save_file(eval_prompt_used, selection_dir, selection_filename.replace(".json", "_gemini_eval_prompt.txt"))
    save_file(eval_text, selection_dir, selection_filename.replace(".json", "_gemini_eval_response.txt"))

    try:
        eval_result = extract_json_from_response(eval_text)
    except Exception as e:
        print(f"    ⚠️ Evaluation JSON parsing failed; defaulting to A. Error: {e}")
        eval_result = {
            "Decision": "A_BETTER",
            "WeightedScoreA": None,
            "WeightedScoreB": None,
            "parse_error": str(e),
            "raw_response": eval_text,
        }

    decision = str(eval_result.get("Decision", "A_BETTER")).strip().upper()
    if decision == "B_BETTER":
        best_idx = 2
        best_src = video_b_path
    else:
        # Match the original pipeline behavior: COMPARABLE or error defaults to A.
        best_idx = 1
        best_src = video_a_path

    best_path = selection_dir / best_filename
    best_video_output = copy_if_local(best_src, best_path)

    selection = {
        "Decision": decision,
        "best_candidate": best_idx,
        "best_video_path": best_video_output,
        "candidate_paths": candidate_paths,
        "WeightedScoreA": eval_result.get("WeightedScoreA"),
        "WeightedScoreB": eval_result.get("WeightedScoreB"),
        "evaluation": eval_result,
    }
    save_file(selection, selection_dir, selection_filename, True)
    print(f"🏆 Best video: candidate_{best_idx} ({decision}) -> {best_video_output}")
    return selection


# ======================
# CLI
# ======================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two or more complete full-storyboard Veo videos independently, then select the best full video with identity-consistency-aware Gemini evaluation."
    )
    parser.add_argument("-t", "--target", required=True, help="Target folder name under results/{target}/")
    parser.add_argument("--script-json", default=None, help="Override path to bestScript.json")
    parser.add_argument("--storyboard-image", default=None, help="Override path to the full storyboard image")
    parser.add_argument("--storyboard-gcs-uri", default=None, help="Optional GCS URI for the storyboard image, useful for Vertex AI")
    parser.add_argument("--storyboard-mime-type", default=None, choices=["image/png", "image/jpeg"], help="MIME type for --storyboard-gcs-uri")
    parser.add_argument("--prompt-file", default=None, help="Prompt template path; defaults to prompts/video_generation_storyboard.txt")
    parser.add_argument("--output-root", default=None, help="Output directory; defaults to video/pipeline_generations_veo31_storyboard_full_candidates_eval_paced/{target}")

    parser.add_argument("--api-backend", default="auto", choices=["auto", "gemini", "vertex"], help="Use Gemini API key mode, Vertex AI mode, or infer from GOOGLE_GENAI_USE_VERTEXAI")
    parser.add_argument("--api-key", default=None, help="Gemini API key override")
    parser.add_argument("--project", default=None, help="Google Cloud project for Vertex AI mode")
    parser.add_argument("--location", default=None, help="Google Cloud location for Vertex AI mode; defaults to GOOGLE_CLOUD_LOCATION or global")

    parser.add_argument("--model", default=os.getenv("VEO_MODEL", "veo-3.1-generate-preview"), help="Veo model id")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"])
    parser.add_argument("--resolution", default="720p", choices=["720p", "1080p", "4k"], help="Initial video resolution. If --extend-steps > 0, this is forced to 720p because extension expects 720p input.")
    parser.add_argument("--duration-seconds", type=int, default=REFERENCE_IMAGE_DURATION_SECONDS, choices=[4, 6, 8])
    parser.add_argument("--person-generation", default="allow_adult", help="Usually allow_adult for image/reference-image generation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-gcs-uri", default=None, help="Optional GCS output prefix for Vertex AI")

    parser.add_argument("--initial-candidates", type=int, default=None, help="Deprecated alias. Use --full-video-candidates instead.")
    parser.add_argument("--extension-candidates", type=int, default=None, help="Deprecated alias. Use --full-video-candidates instead.")
    parser.add_argument("--full-video-candidates", type=int, default=2, help="Number of complete from-start-to-end full video candidates to generate before final A/B scoring")
    parser.add_argument("--extend-steps", type=int, default=1, help="Number of Veo extension steps per full-video candidate. Default 1 implements a paced two-segment plan: panels 1-3 in the initial segment, panels 4-6 in the extension.")
    parser.add_argument("--select-best", action=argparse.BooleanOptionalAction, default=True, help="Evaluate candidate_1 vs candidate_2 and copy the winner")
    parser.add_argument("--eval-model", default=os.getenv("GEMINI_EVAL_MODEL", "gemini-3.1-pro-preview"), help="Gemini model used to evaluate candidate videos")
    parser.add_argument("--compact-script", action=argparse.BooleanOptionalAction, default=True, help="Keep only video-relevant JSON fields in the prompt")
    parser.add_argument("--max-image-mb", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


# ======================
# Main
# ======================
def main() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv()

    args = parse_args()

    # Backward compatibility: if the user still passes --initial-candidates, treat it
    # as the requested number of complete full-video candidates. In this full-candidate
    # pipeline, we do NOT select best at each stage.
    if args.initial_candidates is not None:
        args.full_video_candidates = args.initial_candidates
    if args.extension_candidates is not None and args.extension_candidates != args.full_video_candidates:
        print(
            "⚠️ --extension-candidates is ignored in full-video-candidate mode. "
            "Each full candidate gets exactly one extension chain. Use --full-video-candidates instead."
        )

    if args.full_video_candidates < 1:
        raise ValueError("--full-video-candidates must be >= 1")
    if args.extend_steps < 0:
        raise ValueError("--extend-steps must be >= 0")

    if args.extend_steps > 0 and args.resolution != "720p":
        print("⚠️ Veo extension expects 720p input; forcing --resolution=720p for the initial generation.")
        args.resolution = "720p"

    paths = build_paths(args)
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("===================================================")
    print("🎬 Veo 3.1 full-storyboard FULL-CANDIDATE generation + paced 2-segment scoring")
    print("===================================================")
    print(f"target                : {args.target}")
    print(f"script                : {paths['script_path']}")
    print(f"storyboard            : {paths['storyboard_path']}")
    print(f"prompt file           : {paths['prompt_file']}")
    print(f"output/run            : {run_dir}")
    print(f"model                 : {args.model}")
    print(f"initial duration      : {REFERENCE_IMAGE_DURATION_SECONDS}s (required for reference images)")
    print(f"extension steps/video : {args.extend_steps}")
    print(f"full video candidates : {args.full_video_candidates}")
    print(f"aspect/res            : {args.aspect_ratio}, {args.resolution}")

    with paths["script_path"].open("r", encoding="utf-8") as f:
        full_json = json.load(f)

    initial_prompt = build_storyboard_prompt(
        full_json=full_json,
        prompt_file=paths["prompt_file"],
        use_compact_script=args.compact_script,
    )

    save_file(initial_prompt, run_dir, "veo31_storyboard_initial_prompt.txt")
    save_file(
        {
            "target": args.target,
            "script_path": str(paths["script_path"]),
            "storyboard_path": str(paths["storyboard_path"]),
            "storyboard_gcs_uri": args.storyboard_gcs_uri,
            "model": args.model,
            "aspect_ratio": args.aspect_ratio,
            "resolution": args.resolution,
            "initial_duration_seconds": REFERENCE_IMAGE_DURATION_SECONDS,
            "extension_duration_seconds": EXTENSION_DURATION_SECONDS,
            "full_video_candidates": args.full_video_candidates,
            "extend_steps_per_candidate": args.extend_steps,
            "select_best": args.select_best,
            "eval_model": args.eval_model,
            "mode": "full_candidate_then_final_eval_paced_2_segments",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        run_dir,
        "run_config.json",
        True,
    )

    client = make_genai_client(args)
    reference_image = load_reference_image(args, paths["storyboard_path"])

    full_candidate_results: list[Dict[str, Any]] = []
    full_candidate_summaries: list[Dict[str, Any]] = []

    # Generate complete full-video candidates independently. We intentionally do not
    # select a best initial segment or best continuation segment. This lets the final
    # evaluator judge each complete generated video as a whole, which is better for
    # within-video identity consistency.
    for full_idx in range(1, args.full_video_candidates + 1):
        print("===================================================")
        print(f"🎞️ Generating complete full_candidate_{full_idx}/{args.full_video_candidates}")
        print("    Plan: initial segment targets panels 1-3; extension step 1 targets panels 4-6.")

        candidate_summary: Dict[str, Any] = {
            "full_candidate": full_idx,
            "steps": [],
        }

        initial_result = generate_initial_candidate(
            client=client,
            args=args,
            prompt=initial_prompt,
            reference_image=reference_image,
            run_dir=run_dir,
            candidate_index=full_idx,
        )
        current_result = initial_result
        current_path = str(initial_result["path"])
        current_video_obj = initial_result["video_obj"]
        candidate_summary["steps"].append(
            {"step": "initial", "video_path": current_path}
        )

        for step_index in range(1, args.extend_steps + 1):
            print("---------------------------------------------------")
            print(f"🧩 full_candidate_{full_idx}: extension step {step_index}/{args.extend_steps}")
            extension_prompt = build_extension_prompt(
                full_json=full_json,
                base_prompt=initial_prompt,
                step_index=step_index,
                total_steps=args.extend_steps,
            )
            step_dir = run_dir / f"extend_step_{step_index}"
            save_file(extension_prompt, step_dir, f"veo31_storyboard_extension_step_{step_index}_prompt.txt")

            extension_result = generate_extension_candidate(
                client=client,
                args=args,
                prompt=extension_prompt,
                input_video_obj=current_video_obj,
                run_dir=run_dir,
                step_index=step_index,
                candidate_index=full_idx,
            )
            current_result = extension_result
            current_path = str(extension_result["path"])
            current_video_obj = extension_result["video_obj"]
            candidate_summary["steps"].append(
                {"step": f"extend_step_{step_index}", "video_path": current_path}
            )

        final_candidate_path = copy_if_local(
            current_path,
            run_dir / "full_candidates" / f"full_candidate_{full_idx}.mp4",
        )
        full_result = {
            "path": final_candidate_path,
            "video_obj": current_video_obj,
            "source_path": current_path,
            "full_candidate": full_idx,
        }
        full_candidate_results.append(full_result)
        candidate_summary["final_full_candidate_path"] = final_candidate_path
        full_candidate_summaries.append(candidate_summary)
        save_file(candidate_summary, run_dir / "full_candidates", f"full_candidate_{full_idx}_summary.json", True)
        print(f"✅ full_candidate_{full_idx} complete: {final_candidate_path}")

    save_file(
        {
            "mode": "full_candidate_then_final_eval_paced_2_segments",
            "full_candidates": full_candidate_summaries,
            "candidate_paths": [r["path"] for r in full_candidate_results],
        },
        run_dir / "full_candidates",
        "full_candidate_results.json",
        True,
    )

    if args.select_best:
        print("===================================================")
        print("🏁 Evaluating COMPLETE full videos and selecting best final video")
        final_selection = choose_best_video(
            client=client,
            args=args,
            prompt=initial_prompt,
            candidate_results=full_candidate_results,
            selection_dir=run_dir / "final_selection",
            best_filename="bestFullVideo.mp4",
            selection_filename="best_full_video_selection.json",
            prev_video_path=None,
        )
        best_full_idx = int(final_selection.get("best_candidate", 1))
        selected_path = str(final_selection.get("best_video_path", full_candidate_results[best_full_idx - 1]["path"]))
    else:
        final_selection = {
            "Decision": "SELECTION_DISABLED",
            "best_candidate": 1,
            "best_video_path": full_candidate_results[0]["path"],
        }
        best_full_idx = 1
        selected_path = str(full_candidate_results[0]["path"])

    final_path = copy_if_local(selected_path, run_dir / "finalFullVideo.mp4")
    save_file(
        {
            "final_full_video": final_path,
            "best_full_candidate": best_full_idx,
            "selection": final_selection,
            "full_candidates": full_candidate_summaries,
            "note": "Each candidate is generated from the storyboard and extended independently from beginning to end. The final evaluator compares only the completed full videos.",
        },
        run_dir,
        "final_summary.json",
        True,
    )

    print("===================================================")
    print("DONE")
    for summary in full_candidate_summaries:
        print(f"full_candidate_{summary['full_candidate']}: {summary['final_full_candidate_path']}")
    print(f"best_full_candidate: {best_full_idx}")
    print(f"final_full_video   : {final_path}")
    print(f"run_dir            : {run_dir}")


if __name__ == "__main__":
    main()
