import os
import re
import time
from pathlib import Path

from google import genai

from src.log import error, status, warn

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def make_genai_client() -> genai.Client:
    """Shared Gemini client factory, used by every module that calls Gemini
    (character description, script, storyboard, and reviewer stages).

    Prefers Vertex AI (GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION) when a GCP
    project is configured — this is the team default (most of the team runs
    against Vertex, not a personal API key). Falls back to a plain API key
    (GEMINI_API_KEY) when GOOGLE_CLOUD_PROJECT is unset/empty — that path needs
    nothing more than a key from https://aistudio.google.com/apikey, no GCP IAM/
    service-account/billing setup, and stays available for anyone without GCP
    project access.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        return genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        return genai.Client(api_key=api_key)
    raise RuntimeError(
        "No Gemini credentials found: set GOOGLE_CLOUD_PROJECT (Vertex AI) or "
        "GEMINI_API_KEY (https://aistudio.google.com/apikey) in .env."
    )


def find_character_ref_images(directory: Path) -> list[tuple[int, Path]]:
    """Character reference images for one target directory.

    New convention: <directory>/characters/char{id}.{png,jpg,jpeg}, one image per
    character, returned sorted by id. Falls back to the legacy single
    <directory>/ref.* file (treated as id 1) when characters/ doesn't exist or is
    empty, so existing targets (DualJustice1, DualJustice2, ToutaMairimasu) keep
    working unmodified. Raises FileNotFoundError if neither exists.
    """
    chars_dir = directory / "characters"
    if chars_dir.is_dir():
        pairs: list[tuple[int, Path]] = []
        for p in chars_dir.iterdir():
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            m = re.match(r"char(\d+)$", p.stem)
            if m:
                pairs.append((int(m.group(1)), p))
        if pairs:
            pairs.sort(key=lambda t: t[0])
            return pairs
    matches = list(directory.glob("ref.*"))
    if matches:
        return [(1, matches[0])]
    raise FileNotFoundError(
        f"No character reference image found: expected {directory}/characters/char<id>.* "
        f"or a legacy {directory}/ref.*"
    )


def with_retry(fn, max_retries: int = 20, delay: float = 5.0, default=None, label: str | None = None):
    """Call fn(), retrying up to max_retries times on any exception.

    If `label` is given, an animated spinner with that label runs during each
    attempt. Returns `default` if all attempts fail instead of raising.
    """
    for attempt in range(1, max_retries + 1):
        try:
            if label:
                with status(f"{label} (attempt {attempt}/{max_retries})"):
                    return fn()
            return fn()
        except Exception as e:
            if attempt == max_retries:
                error(f"All {max_retries} attempts failed: {e}. Skipping.")
                return default
            warn(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
