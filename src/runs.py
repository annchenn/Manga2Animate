"""Helpers for the timestamped run folders the pipeline writes into.

Layout:
    <results-root>/<target>/<YYYYmmdd_HHMMSS>/   one pipeline run (magi, character, iter*)
    <results-root>/<target>/video/              video output, NOT per-run

`run_all.py` picks a run id and exports it as RUN_ID; the pipeline modules read it
back through `results_dir_from_env`. Tools that consume a finished run (video
generation, panel cropping) resolve it with `resolve_run_dir`, which defaults to
the newest run when no timestamp is given.
"""

import os
import re
from datetime import datetime
from pathlib import Path

RUN_ID_RE = re.compile(r"^\d{8}_\d{6}$")


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_run_dirs(target_dir: Path) -> list[Path]:
    """All timestamped run folders under a target, oldest first."""
    if not target_dir.is_dir():
        return []
    return sorted(
        (d for d in target_dir.iterdir() if d.is_dir() and RUN_ID_RE.match(d.name)),
        key=lambda d: d.name,
    )


def latest_run_dir(target_dir: Path) -> Path | None:
    runs = list_run_dirs(target_dir)
    return runs[-1] if runs else None


def resolve_run_dir(target_dir: Path, run_id: str | None, allow_legacy: bool = True) -> Path:
    """Return <target_dir>/<run_id>, or the newest run folder when run_id is None.

    With allow_legacy, a target that has no timestamped folders at all (results
    written before this layout) resolves to the target directory itself.
    """
    if run_id:
        run_dir = target_dir / run_id
        if not run_dir.is_dir():
            available = [d.name for d in list_run_dirs(target_dir)]
            raise FileNotFoundError(
                f"Run '{run_id}' not found under {target_dir}. Available: {available or 'none'}"
            )
        return run_dir

    latest = latest_run_dir(target_dir)
    if latest is not None:
        return latest
    if allow_legacy and target_dir.is_dir():
        return target_dir
    raise FileNotFoundError(f"No timestamped run directory under {target_dir}")


def results_dir_from_env(base_dir: Path) -> Path:
    """RESULTS_DIR for a pipeline module, honouring TARGET / RESULTS_ROOT / RUN_ID.

    Without RUN_ID (module run standalone) this falls back to the newest run,
    matching "no --time means the latest one".
    """
    target = os.environ.get("TARGET", "")
    results_root = os.environ.get("RESULTS_ROOT", "results")
    target_dir = base_dir / results_root / target if target else base_dir / results_root

    run_id = os.environ.get("RUN_ID", "")
    if run_id:
        return target_dir / run_id
    return resolve_run_dir(target_dir, None)
