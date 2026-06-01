#!/usr/bin/env python3
"""
Generate two continuous videos from a complete storyboard image using Veo 3.1, then use Gemini to select the better one.

This script is intentionally different from video/pipeline.py:
- video/pipeline.py splits the storyboard into per-panel images and generates clips independently.
- This script sends the full bestStoryboard.png as a single Veo 3.1 reference image, so the model can use the whole storyboard as visual continuity guidance.
- It generates two independent candidates and uses Gemini to pick the better one with the original first-video Manga2Animate rubric.
- It does not use video extension.

Typical usage from the Manga2Animate project root:

    python video/pipeline_veo31_storyboard_two_eval.py -t death_notebook

Useful overrides:

    python video/pipeline_veo31_storyboard_two_eval.py \
        -t death_notebook \
        --storyboard-image results/death_notebook/iter1/bestStoryboard.png \
        --aspect-ratio 16:9 \
        --resolution 720p

Authentication:
- Gemini API key mode: set GEMINI_API_KEY or GOOGLE_API_KEY.
- Vertex AI mode: set GOOGLE_GENAI_USE_VERTEXAI=True, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import os
import re
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "video" / "pipeline_generations_veo31_storyboard_two_eval"

# Veo 3.1 reference-image generation currently requires an 8-second duration.
REFERENCE_IMAGE_DURATION_SECONDS = 8


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
    """
    Keep only the fields that are useful for video generation.

    Veo has a relatively small text prompt budget, so this avoids sending reviewer
    metadata or unrelated fields if future script JSON files contain them.
    """
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

Additional Veo 3.1 full-storyboard reference rules:
- Treat the reference image as the complete storyboard for the whole video, not as a single still frame to reproduce.
- Use the panels as ordered visual beats from left-to-right, top-to-bottom unless the script implies another order.
- Animate the story as one continuous sequence with smooth motion between beats; do not generate six independent clips stitched together.
- Preserve character design, style, composition cues, action direction, and spatial continuity from the storyboard.
- Do not show panel borders, gutters, captions, speech bubbles, dialogue text, or sound-effect text in the output video.
- When a subject is moving across panels, infer the in-between motion instead of abruptly teleporting or resetting the subject.
""".strip()

    return f"{prompt.strip()}\n\n{storyboard_specific_rules}\n"


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
    # Veo reference images currently document png/jpeg support most consistently.
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

    # types.Image.from_file infers MIME type from the local path.
    return types.Image.from_file(location=str(storyboard_path))


def build_generate_config(args: argparse.Namespace, reference_image) -> Any:
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


def save_generated_video(client, operation, output_path: Path) -> Optional[str]:
    if not getattr(operation, "response", None):
        raise RuntimeError(f"Veo operation finished without response: {operation}")

    generated_videos = getattr(operation.response, "generated_videos", None)
    if not generated_videos:
        filtered_count = getattr(operation.response, "rai_media_filtered_count", None)
        filtered_reasons = getattr(operation.response, "rai_media_filtered_reasons", None)
        if filtered_count:
            raise RuntimeError(
                "Veo generated no video because the output was filtered by Google's guardrails.\n"
                f"rai_media_filtered_count={filtered_count}\n"
                f"rai_media_filtered_reasons={filtered_reasons}\n"
                "This is usually caused by prompt/reference content being too close to third-party IP, "
                "or by other safety/memorization filters. Try original/licensed assets and remove franchise names."
            )
        raise RuntimeError(f"Veo operation response contains no generated_videos: {operation.response}")

    video_obj = generated_videos[0].video

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
        return str(output_path)

    if uri:
        return uri

    raise RuntimeError("Could not save generated video locally and no video URI was returned.")


def generate_one_candidate(
    client,
    args: argparse.Namespace,
    prompt: str,
    reference_image,
    run_dir: Path,
    candidate_index: int,
) -> str:
    candidate_dir = run_dir / f"candidate_{candidate_index}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    output_path = candidate_dir / f"veo31_storyboard_candidate_{candidate_index}.mp4"
    if output_path.exists() and args.auto_resume:
        print(f"⏭️ candidate_{candidate_index} already exists: {output_path}")
        return str(output_path)

    config = build_generate_config(args, reference_image)
    save_file(model_to_jsonable(config), candidate_dir, "veo_config.json", True)

    print(f"🚀 Submitting candidate_{candidate_index} to Veo 3.1: model={args.model}")
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

    saved_path_or_uri = save_generated_video(client, operation, output_path)
    print(f"✅ candidate_{candidate_index} saved/result: {saved_path_or_uri}")
    return saved_path_or_uri


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


def extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a Gemini response."""
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError(f"Could not find a JSON object in Gemini response:\n{text}")


def evaluate_two_videos_original_first_rules(
    client,
    user_prompt: str,
    video_a_path: str,
    video_b_path: str,
    eval_model: str,
    run_dir: Path,
) -> Dict[str, Any]:
    """Compare two videos using the original first-video Manga2Animate-style rubric."""
    file_a = upload_to_gemini_for_eval(client, video_a_path)
    file_b = upload_to_gemini_for_eval(client, video_b_path)

    eval_prompt = f"""You are a very critical and mindful expert video evaluator tasked with comparing two animated videos, A and B, to determine which one is the better generated animation for the User Prompt.

User Prompt:
{user_prompt}

Important Constraints:
- The winning video must better adhere to the User Prompt.
- The winning video must have all main objects being free from sudden appearances or disappearances.
- The winning video must be free from too many unnecessary scene transitions.
- The winning video must have characters' movements free from unnaturally fast or slow speeds.
- The winning video should feel like a coherent animated scene instead of a rushed slideshow of storyboard panels.

For each criterion below, assign a score of 1 (A wins), 0 (B wins), or 0.5 (TIE). Justify your score with concrete video evidence.

Criteria:
1. Visual Realism / Animation Quality (Weight: 0.35)
   Which video has fewer distracting AI artifacts, distorted faces/bodies, flickering, broken objects, sudden visual changes, or unnatural rendering?

2. Physical Commonsense (Weight: 0.30)
   Which video's actions, environmental setting, events, movements, object continuity, and character motion are more internally logical and physically plausible?

3. Video-Prompt Alignment (Weight: 0.35)
   Which video more accurately matches the User Prompt in terms of visuals, actions, story beats, camera intent, constraints, and action direction?

Perform the Following Steps One-by-One:
1. Criterion Evaluation:
   - For each criterion, evaluate A vs. B based on the sub-criteria.
   - For each criterion, assign a score: 1 (A wins), 0 (B wins), 0.5 (TIE).
   - For each criterion, provide a concise explanation with specific evidence from the videos.
2. Weighted Score Calculation:
   - Compute weighted score for A:
     WeightedScoreA = 0.35 * VisualRealismScoreA + 0.30 * PhysicalCommonsenseScoreA + 0.35 * VideoPromptAlignmentScoreA
   - Since each criterion score is A-vs-B, use:
     A receives criterion score s, B receives 1-s.
3. Final Decision:
   - If |WeightedScoreA - WeightedScoreB| < 0.05, output COMPARABLE.
   - Otherwise, output A_BETTER if WeightedScoreA > WeightedScoreB, or B_BETTER if WeightedScoreB > WeightedScoreA.

Output strictly as a JSON block:
```json
{{
  "Reasoning_VisualRealism": "<concise explanation>",
  "Reasoning_PhysicalCommonsense": "<concise explanation>",
  "Reasoning_VideoPromptAlignment": "<concise explanation>",
  "VisualRealism_A_vs_B": <1 | 0.5 | 0>,
  "PhysicalCommonsense_A_vs_B": <1 | 0.5 | 0>,
  "VideoPromptAlignment_A_vs_B": <1 | 0.5 | 0>,
  "WeightedScoreA": <float>,
  "WeightedScoreB": <float>,
  "Decision": "<A_BETTER | B_BETTER | COMPARABLE>",
  "BetterCandidate": "<candidate_1 | candidate_2 | comparable>"
}}
```
"""

    save_file(eval_prompt, run_dir, "gemini_eval_prompt.txt")

    response = client.models.generate_content(
        model=eval_model,
        contents=["Video A:", file_a, "Video B:", file_b, eval_prompt],
    )
    response_text = getattr(response, "text", None) or str(response)
    save_file(response_text, run_dir, "gemini_eval_response.txt")

    parsed = extract_json_object(response_text)
    decision = str(parsed.get("Decision", "")).strip().upper()

    if decision == "A_BETTER":
        best_index = 1
    elif decision == "B_BETTER":
        best_index = 2
    else:
        # Keep the output deterministic when Gemini considers the videos comparable.
        best_index = 1
        parsed["Decision"] = "COMPARABLE"
        parsed["TieBreakRule"] = "Videos were comparable; selected candidate_1 by default for deterministic output."

    parsed["best_candidate_index"] = best_index
    parsed["best_video_path"] = video_a_path if best_index == 1 else video_b_path

    save_file(parsed, run_dir, "best_video_selection.json", True)
    return parsed


def copy_best_video(best_video_path: str, run_dir: Path) -> Dict[str, str]:
    src = Path(best_video_path)
    if not src.exists():
        raise FileNotFoundError(f"Selected best video is not a local file and cannot be copied: {best_video_path}")

    best_path = run_dir / "bestVideo.mp4"
    final_path = run_dir / "finalVideo.mp4"
    shutil.copy2(src, best_path)
    shutil.copy2(src, final_path)
    return {"bestVideo": str(best_path), "finalVideo": str(final_path)}


# ======================
# CLI
# ======================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate two full-storyboard Veo 3.1 candidates and use Gemini to select the better one."
    )
    parser.add_argument("-t", "--target", required=True, help="Target folder name under results/{target}/")
    parser.add_argument("--script-json", default=None, help="Override path to bestScript.json")
    parser.add_argument("--storyboard-image", default=None, help="Override path to the full storyboard image")
    parser.add_argument("--storyboard-gcs-uri", default=None, help="Optional GCS URI for the storyboard image, useful for Vertex AI")
    parser.add_argument("--storyboard-mime-type", default=None, choices=["image/png", "image/jpeg"], help="MIME type for --storyboard-gcs-uri")
    parser.add_argument("--prompt-file", default=None, help="Prompt template path; defaults to prompts/video_generation_storyboard.txt")
    parser.add_argument("--output-root", default=None, help="Output directory; defaults to video/pipeline_generations_veo31_storyboard_two_eval/{target}")

    parser.add_argument("--api-backend", default="auto", choices=["auto", "gemini", "vertex"], help="Use Gemini API key mode, Vertex AI mode, or infer from GOOGLE_GENAI_USE_VERTEXAI")
    parser.add_argument("--api-key", default=None, help="Gemini API key override")
    parser.add_argument("--project", default=None, help="Google Cloud project for Vertex AI mode")
    parser.add_argument("--location", default=None, help="Google Cloud location for Vertex AI mode; defaults to GOOGLE_CLOUD_LOCATION or global")

    parser.add_argument("--model", default=os.getenv("VEO_MODEL", "veo-3.1-generate-preview"), help="Veo model id")
    parser.add_argument("--aspect-ratio", default="16:9", choices=["16:9", "9:16"])
    parser.add_argument("--resolution", default="720p", choices=["720p", "1080p", "4k"])
    parser.add_argument("--duration-seconds", type=int, default=REFERENCE_IMAGE_DURATION_SECONDS, choices=[4, 6, 8])
    parser.add_argument("--person-generation", default="allow_adult", help="Usually allow_adult for image/reference-image generation")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-gcs-uri", default=None, help="Optional GCS output prefix for Vertex AI")

    parser.add_argument("--candidates", type=int, default=2, help="Submit N separate Veo operations. Default 2 for automatic A/B comparison")
    parser.add_argument("--select-best", action=argparse.BooleanOptionalAction, default=True, help="Use Gemini to compare candidate_1 and candidate_2, then copy the winner to bestVideo.mp4/finalVideo.mp4")
    parser.add_argument("--eval-model", default=os.getenv("GEMINI_EVAL_MODEL", "gemini-3.1-pro-preview"), help="Gemini model used to evaluate candidate videos")
    parser.add_argument("--compact-script", action=argparse.BooleanOptionalAction, default=True, help="Keep only video-relevant JSON fields in the prompt")
    parser.add_argument("--max-image-mb", type=float, default=20.0)
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--auto-resume", action="store_true", help="Reuse the latest run folder and skip existing local candidate videos")
    return parser.parse_args()


# ======================
# Main
# ======================
def main() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv()

    args = parse_args()
    if args.candidates < 1:
        raise ValueError("--candidates must be >= 1")
    if args.select_best and args.candidates < 2:
        raise ValueError("--select-best requires --candidates >= 2")

    paths = build_paths(args)
    output_root = paths["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    if args.auto_resume:
        existing_runs = sorted(p for p in output_root.glob("run_*") if p.is_dir())
        run_dir = existing_runs[-1] if existing_runs else None
    else:
        run_dir = None

    if run_dir is None:
        run_dir = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

    print("======================================")
    print("🎬 Veo 3.1 full-storyboard generation + Gemini A/B selection")
    print("======================================")
    print(f"target        : {args.target}")
    print(f"script        : {paths['script_path']}")
    print(f"storyboard    : {paths['storyboard_path']}")
    print(f"prompt file   : {paths['prompt_file']}")
    print(f"output/run    : {run_dir}")
    print(f"model         : {args.model}")
    print(f"duration      : {REFERENCE_IMAGE_DURATION_SECONDS}s (required for reference images)")
    print(f"aspect/res    : {args.aspect_ratio}, {args.resolution}")
    print(f"candidates    : {args.candidates}")
    print(f"select best   : {args.select_best}")
    print(f"eval model    : {args.eval_model}")

    with paths["script_path"].open("r", encoding="utf-8") as f:
        full_json = json.load(f)

    prompt = build_storyboard_prompt(
        full_json=full_json,
        prompt_file=paths["prompt_file"],
        use_compact_script=args.compact_script,
    )

    save_file(prompt, run_dir, "veo31_storyboard_prompt.txt")
    save_file(
        {
            "target": args.target,
            "script_path": str(paths["script_path"]),
            "storyboard_path": str(paths["storyboard_path"]),
            "storyboard_gcs_uri": args.storyboard_gcs_uri,
            "model": args.model,
            "aspect_ratio": args.aspect_ratio,
            "resolution": args.resolution,
            "duration_seconds": REFERENCE_IMAGE_DURATION_SECONDS,
            "candidates": args.candidates,
            "select_best": args.select_best,
            "eval_model": args.eval_model,
            "mode": "two_candidates_then_gemini_original_first_rules",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        run_dir,
        "run_config.json",
        True,
    )

    client = make_genai_client(args)
    reference_image = load_reference_image(args, paths["storyboard_path"])

    results = []
    for candidate_index in range(1, args.candidates + 1):
        result = generate_one_candidate(
            client=client,
            args=args,
            prompt=prompt,
            reference_image=reference_image,
            run_dir=run_dir,
            candidate_index=candidate_index,
        )
        results.append(result)

    final_summary: Dict[str, Any] = {"results": results}

    if args.select_best:
        print("======================================")
        print("🔎 Evaluating candidate_1 vs candidate_2 with Gemini...")
        selection = evaluate_two_videos_original_first_rules(
            client=client,
            user_prompt=prompt,
            video_a_path=results[0],
            video_b_path=results[1],
            eval_model=args.eval_model,
            run_dir=run_dir,
        )
        copied = copy_best_video(selection["best_video_path"], run_dir)
        final_summary["selection"] = selection
        final_summary["copied_outputs"] = copied

    save_file(final_summary, run_dir, "results.json", True)
    print("======================================")
    print("DONE")
    for idx, result in enumerate(results, start=1):
        print(f"candidate_{idx}: {result}")

    if args.select_best:
        selection = final_summary["selection"]
        print("--------------------------------------")
        print(f"Decision      : {selection.get('Decision')}")
        print(f"WeightedScoreA: {selection.get('WeightedScoreA')}")
        print(f"WeightedScoreB: {selection.get('WeightedScoreB')}")
        print(f"Best candidate: candidate_{selection.get('best_candidate_index')}")
        print(f"bestVideo     : {final_summary['copied_outputs']['bestVideo']}")
        print(f"finalVideo    : {final_summary['copied_outputs']['finalVideo']}")

    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
