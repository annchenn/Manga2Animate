"""Per-run metrics: how long each stage took, and how many Gemini calls it made.

Answers two questions about a single example (one target, one run):
  - wall-clock time to produce a complete storyboard, broken down by stage;
  - number of Gemini API calls, which is what the run actually costs.

Collection points are deliberately few. Every Gemini-calling module in the
storyboard pipeline gets its client from src.utils.make_genai_client(), so
wrapping the client there (instrument_client) covers all of them with no
per-module instrumentation. src/create_video.py constructs genai.Client directly
and is therefore NOT counted -- correct, since that is the separate video stage.

Two call counts are tracked because they differ, and only one of them is money:

  api_calls      requests actually sent, retries included. src.utils.with_retry
                 retries up to 20 times on any exception, so one logical step can
                 be many billed requests.
  logical_calls  pipeline-level steps (one per with_retry block).

`failures` matters for the same reason: with_retry returns its `default` instead of
raising when every attempt fails, so a run that looks clean can contain calls that
never succeeded. Counting them keeps that visible.
"""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

METRICS_FILENAME = "metrics.json"
LOG_FILENAME = "metrics_log.jsonl"

# Stage label used for calls made outside any stage() block (e.g. a module-level
# client call), so they are still counted rather than silently dropped.
_UNSCOPED = "(unscoped)"


class _Collector:
    def __init__(self):
        self.started_at = time.time()
        self.current_stage = _UNSCOPED
        self.stage_seconds: dict[str, float] = {}
        self.stages_skipped: list[str] = []
        self.api_calls = 0
        self.logical_calls = 0
        self.retries = 0
        self.failures = 0
        self.by_stage: dict[str, int] = {}
        self.by_model: dict[str, int] = {}
        self.extra: dict[str, object] = {}

    # -- timing ----------------------------------------------------------------
    @contextmanager
    def stage(self, name: str):
        """Time a stage and attribute any Gemini calls inside it to `name`."""
        previous = self.current_stage
        self.current_stage = name
        t0 = time.time()
        try:
            yield
        finally:
            self.stage_seconds[name] = round(self.stage_seconds.get(name, 0.0) + time.time() - t0, 2)
            self.current_stage = previous

    def mark_skipped(self, name: str) -> None:
        """Record a stage that auto-resume skipped.

        Kept separate from a 0.0-second entry on purpose: a skipped stage did no
        work, and folding it in as "0 seconds" would make a resumed run look
        dramatically faster than a real one.
        """
        if name not in self.stages_skipped:
            self.stages_skipped.append(name)

    # -- Gemini accounting -----------------------------------------------------
    def record_attempt(self, model: str | None) -> None:
        self.api_calls += 1
        key = self.current_stage
        self.by_stage[key] = self.by_stage.get(key, 0) + 1
        if model:
            self.by_model[model] = self.by_model.get(model, 0) + 1

    def record_logical(self) -> None:
        self.logical_calls += 1

    def record_retry(self) -> None:
        self.retries += 1

    def record_failure(self) -> None:
        self.failures += 1

    def add(self, key: str, value) -> None:
        """Attach a sub-document (e.g. the flux2 worker's own timings)."""
        self.extra[key] = value

    # -- output ----------------------------------------------------------------
    def payload(self, target: str | None = None, run_id: str | None = None) -> dict:
        return {
            "target": target or os.environ.get("TARGET", ""),
            "run_id": run_id or os.environ.get("RUN_ID", ""),
            "storyboard_backend": os.environ.get("STORYBOARD_BACKEND", "gemini"),
            "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "wall_clock_s": round(time.time() - self.started_at, 2),
            "stages": dict(self.stage_seconds),
            "stages_skipped": list(self.stages_skipped),
            "gemini": {
                "api_calls": self.api_calls,
                "logical_calls": self.logical_calls,
                "retries": self.retries,
                "failures": self.failures,
                "by_stage": dict(self.by_stage),
                "by_model": dict(self.by_model),
            },
            **self.extra,
        }


_collector = _Collector()


def collector() -> _Collector:
    return _collector


def reset() -> None:
    """Fresh collector. For tests; a pipeline process only ever needs one."""
    global _collector
    _collector = _Collector()


def stage(name: str):
    return _collector.stage(name)


def mark_skipped(name: str) -> None:
    _collector.mark_skipped(name)


def add(key: str, value) -> None:
    _collector.add(key, value)


def _model_of(kwargs: dict, args: tuple) -> str | None:
    if "model" in kwargs:
        return str(kwargs["model"])
    return str(args[0]) if args else None


class _CountingModels:
    """Wraps client.models, counting generate_content / generate_content_stream.

    Only the two generation entry points are intercepted; everything else is
    passed through untouched, so this stays transparent to callers.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name not in {"generate_content", "generate_content_stream"}:
            return attr

        def wrapper(*args, **kwargs):
            _collector.record_attempt(_model_of(kwargs, args))
            return attr(*args, **kwargs)

        return wrapper


class _CountingClient:
    def __init__(self, inner):
        self._inner = inner
        self.models = _CountingModels(inner.models)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def instrument_client(client):
    """Return `client` with its generation calls counted."""
    return _CountingClient(client)


def write_metrics(run_dir: Path, target: str | None = None, run_id: str | None = None) -> Path:
    """Write metrics.json into the run dir, and append one row to the shared log.

    The JSONL log is append-only (same pattern as the batch driver's timing_log)
    so durations and call counts accumulate across every example and can be
    compared without re-reading each run folder.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = _collector.payload(target, run_id)

    path = run_dir / METRICS_FILENAME
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log_path = run_dir.parent.parent / LOG_FILENAME
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except OSError:
        pass  # the per-run metrics.json is the source of truth; the log is a convenience
    return path
