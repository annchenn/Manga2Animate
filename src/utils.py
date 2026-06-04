import time

from src.log import error, status, warn


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
