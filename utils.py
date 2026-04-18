import time


def with_retry(fn, max_retries: int = 3, delay: float = 5.0, default=None):
    """Call fn(), retrying up to max_retries times on any exception.
    Returns default if all attempts fail instead of raising."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries:
                print(f"  All {max_retries} attempts failed: {e}. Skipping.")
                return default
            print(f"  Attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
