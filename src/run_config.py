"""Record which models a pipeline run used.

`run_all.py` writes this into `<results-root>/<target>/<run timestamp>/config.json`,
next to `log.txt`.

Storyboard models only. Video generation is a separate stage with its own run
folders under `<target>/video_fullstoryboard/<model>/run_*`, so `VIDEO_*` from
`.env` says nothing about what this run did and is deliberately left out.
"""

import json
import os
from datetime import datetime
from pathlib import Path

CONFIG_FILENAME = "config.json"


def collect_models() -> dict:
    """The model ids from .env, as the pipeline resolved them."""
    return {
        "text": os.environ.get("TEXT_MODEL", "gemini-3.1-pro-preview"),
        "image": os.environ.get("IMAGE_MODEL", "gemini-3-pro-image"),
    }


def write_run_config(run_dir: Path, run_id: str, target: str) -> tuple[Path, dict | None]:
    """Write config.json. Returns the path and the previous models, if any.

    A resumed run rewrites the file, so the caller gets the old models back and
    can say something when `.env` changed between invocations.
    """
    path = run_dir / CONFIG_FILENAME

    previous = None
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("models")
        except (OSError, json.JSONDecodeError):
            previous = None

    payload = {
        "run_id": run_id,
        "target": target,
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "models": collect_models(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, previous
