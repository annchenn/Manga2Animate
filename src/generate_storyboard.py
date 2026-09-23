import json
import mimetypes
import os
from pathlib import Path
from google.genai import types
from dotenv import load_dotenv
from src.log import error, info, status, step, subsection, success, warn
from src.runs import results_dir_from_env
from src.utils import find_character_ref_images, make_genai_client, with_retry

load_dotenv()

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
_target = os.environ.get("TARGET", "")
_target_root = os.environ.get("TARGET_ROOT", "target")
TARGET_DIR = BASE_DIR / _target_root / _target if _target else BASE_DIR / _target_root
RESULTS_DIR = results_dir_from_env(BASE_DIR)  # <results-root>/<target>/<run timestamp>
PROMPT_FILE = BASE_DIR / "prompts" / "create_storyboard.txt"

MODEL = os.environ.get("IMAGE_MODEL", "gemini-3-pro-image")
NUM_STORYBOARDS = int(os.environ.get("NUM_STORYBOARDS", "3"))
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "6"))
STORYBOARD_BACKEND = os.environ.get("STORYBOARD_BACKEND", "gemini")

client = make_genai_client()


def iter_dir(iteration: int) -> Path:
    return RESULTS_DIR / f"iter{iteration}"


def build_storyboard_prompt(best_script: dict) -> str:
    template = PROMPT_FILE.read_text(encoding="utf-8")
    template = template.replace(
        "{Target Animation Script}",
        json.dumps(best_script, ensure_ascii=False, indent=2),
    )
    return template


def generate_one(best_script: dict, iteration: int, sb_index: int):
    prompt_text = build_storyboard_prompt(best_script)

    image_paths = [
        *(TARGET_DIR / "manga_panels" / f"panel{i}.png" for i in range(1, NUM_CLIPS + 1)),
        *(p for _, p in find_character_ref_images(TARGET_DIR)),
    ]

    parts = []
    for img_path in image_paths:
        mime_type, _ = mimetypes.guess_type(str(img_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime_type))
    parts.append(types.Part.from_text(text=prompt_text))

    contents = [types.Content(role="user", parts=parts)]
    config = types.GenerateContentConfig(
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="1K",
        ),
        response_modalities=["IMAGE"],
    )

    out_dir = iter_dir(iteration) / "storyboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    def _stream():
        for chunk in client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=config,
        ):
            if chunk.parts is None:
                continue
            part = chunk.parts[0]
            if part.inline_data and part.inline_data.data:
                ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".png"
                out_path = out_dir / f"storyboard_{sb_index}{ext}"
                out_path.write_bytes(part.inline_data.data)
                saved_paths.append(out_path)

    with_retry(_stream, label=f"Generating storyboard {sb_index} ({MODEL})")
    for p in saved_paths:
        success(f"Saved {p}")


def main(iteration: int = 0):
    best_path = iter_dir(iteration) / "bestScript.json"
    if not best_path.exists():
        error(f"{best_path} not found. Run evaluate_scripts first.")
        return
    best_script = json.loads(best_path.read_text(encoding="utf-8"))

    subsection(f"generate_storyboard (iter={iteration}, backend={STORYBOARD_BACKEND})")

    if STORYBOARD_BACKEND == "flux2":
        generate_flux2(best_script, iteration)
        return

    for i in range(NUM_STORYBOARDS):
        step(f"Storyboard {i + 1}/{NUM_STORYBOARDS}")
        generate_one(best_script, iteration, i)


def generate_flux2(best_script: dict, iteration: int) -> None:
    """Open-source storyboard backend: per-panel FLUX.2-dev generation (via a
    cross-env subprocess bridge) + a Gemini best-of-N panel reviewer + local grid
    composition. Produces exactly one storyboard candidate (storyboard_0.png),
    unlike the gemini backend's NUM_STORYBOARDS independent whole-grid candidates."""
    from PIL import Image
    from src import evaluate_panel_candidates, flux2_bridge, grid_compose

    character_description = json.loads(
        (RESULTS_DIR / "character" / "character_description.json").read_text(encoding="utf-8")
    )
    out_dir = iter_dir(iteration) / "storyboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    step("Generating per-panel candidates (flux2)")
    flux2_bridge.generate_candidates(
        best_script, character_description, TARGET_DIR, TARGET_DIR / "manga_panels", out_dir
    )

    step("Reviewing per-panel candidates (Gemini)")
    winners = evaluate_panel_candidates.main(iteration)
    if len(winners) != NUM_CLIPS:
        warn(f"Only {len(winners)}/{NUM_CLIPS} panels have a selected winner")

    step("Composing storyboard grid")
    selected_dir = out_dir / "selected"
    panel_ids = sorted(c["id"] for c in best_script["clips"])
    panels = [Image.open(selected_dir / f"panel{pid}.png").convert("RGB") for pid in panel_ids]
    grid = grid_compose.compose_grid(
        panels, columns=3, cell_size=(flux2_bridge.FLUX2_PANEL_WIDTH, flux2_bridge.FLUX2_PANEL_HEIGHT)
    )
    out_path = out_dir / "storyboard_0.png"
    grid.save(out_path)
    success(f"Saved {out_path}")


if __name__ == "__main__":
    main()
