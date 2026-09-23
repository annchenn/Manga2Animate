import os
import time
import json
import random
import re
import threading
import requests
import io
import base64
from PIL import Image


# Params some OpenAI-compatible endpoints reject (e.g. Gemini rejects 'seed' with
# HTTP 400). Dropped for the rest of the process once an endpoint refuses them.
_UNSUPPORTED_PARAMS = set()


def gptv_query(transcript=None, top_p=0.2, temp=0., model_type="gpt-4.1", api_key='', base_url='', seed=123, max_tokens=512, wait_time=10, max_retries=8):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    base_url = base_url[:-1] if base_url.endswith('/') else base_url
    requests_url = f"{base_url}/chat/completions" if base_url.endswith('/v1') else f"{base_url}/v1/chat/completions"

    data = {
        'model': model_type,
        'max_tokens': max_tokens,
        'temperature': temp,
        'messages': transcript or [],
    }
    if seed is not None and 'seed' not in _UNSUPPORTED_PARAMS:
        data['seed'] = seed

    response_text, retry, response_json, last_error = '', 0, None, 'no attempt made'
    rate_limit_waits = 0
    while len(response_text) < 2 and retry < max_retries:
        retry += 1
        try:
            _rate_limiter.wait()
            response = requests.post(url=requests_url, headers=headers, data=json.dumps(data))
            response_json = response.json()
        except Exception as e:
            last_error = str(e)
            print(e)
            time.sleep(wait_time)
            continue
        if response.status_code != 200:
            body = response.text
            last_error = f"HTTP {response.status_code}: {body[:300]}"
            # An endpoint that rejects an optional param will keep rejecting it, so
            # drop it and retry immediately instead of burning the retry budget.
            dropped = False
            for param in ('seed', 'top_p'):
                if param in data and param in body and (
                    'Unknown name' in body or 'Unrecognized' in body or 'unsupported' in body.lower()
                ):
                    print(f"Endpoint rejected '{param}'; dropping it and retrying.")
                    _UNSUPPORTED_PARAMS.add(param)
                    del data[param]
                    dropped = True
            if dropped:
                retry -= 1
                continue

            # Rate limiting is not a bad request: wait exactly as long as the endpoint
            # asks and try the same call again. Nudging the temperature here (as the
            # generic path below does) would also quietly change what the repeated
            # scoring is sampling, which is the one thing that must stay fixed.
            if response.status_code == 429:
                # Not every 429 is pacing: an exhausted balance or a spent quota comes
                # back with the same status and will never succeed, so retrying it just
                # burns hours the way the Gemini 403 block did on 2026-09-05. Fail the
                # whole run immediately and let the caller surface the billing problem.
                if any(marker in body for marker in
                       ('insufficient_quota', 'credit_balance_exhausted', 'billing_hard_limit_reached')):
                    raise RuntimeError(f"API quota/credit exhausted, not a rate limit: {body[:200]}")
                delay = wait_time
                for header in ('retry-after-ms', 'Retry-After'):
                    value = response.headers.get(header)
                    if value:
                        try:
                            delay = float(value) / (1000.0 if header == 'retry-after-ms' else 1.0)
                        except ValueError:
                            pass
                        break
                delay = min(max(delay + random.uniform(0.2, 1.5), 1.0), 60.0)
                print(f"Rate limited; retrying in {delay:.1f}s")
                # Waiting out this one call is not enough: if the pacing were right the
                # 429 would not have happened, so widen it for every worker as well.
                _rate_limiter.penalize()
                time.sleep(delay)
                rate_limit_waits += 1
                if rate_limit_waits <= 20:
                    retry -= 1
                continue

            print(f"HTTP {response.status_code}: {body[:300]}")
            time.sleep(wait_time)
            data['temperature'] = min(data['temperature'] + 0.2, 1.0)
            continue
        if 'choices' not in response_json:
            last_error = f"no 'choices' in response: {str(response_json)[:300]}"
            time.sleep(wait_time)
            continue
        response_text = response_json["choices"][0]["message"]["content"]
        _rate_limiter.reward()

    if len(response_text) < 2:
        raise RuntimeError(f"gptv_query failed after {max_retries} attempts. Last error: {last_error}")
    return response_text


def encode_image(image_input, image_mode='path', resize_to=(512, 512)):
    if image_mode == 'path':
        with Image.open(image_input) as img:
            img = img.resize(resize_to)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    elif image_mode == 'pil':
        img = image_input.resize(resize_to)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')


def load_img_content(image_input, image_mode='path', resize_to=(512, 512), pil_detail='low'):
    base64_image = encode_image(image_input, image_mode, resize_to=resize_to)
    image_meta = "data:image/jpeg;base64"
    content = {
        "type": "image_url",
        "image_url": {"url": f"{image_meta},{base64_image}"},
    }
    if image_mode == 'pil':
        content["image_url"]["detail"] = pil_detail  # PIL image is small, configurable detail
    return content


SCORE_PATTERN = re.compile(r"(score|Score):\s*[a-zA-Z]*\s*(\d+)")


class _RateLimiter:
    """
    Global minimum spacing between VLM requests, widened automatically on 429.

    Endpoints with a tight tokens-per-minute budget (OpenAI's gpt-4.1 here is 30k TPM,
    about 20 of these image calls a minute) cannot be driven by worker count: 32 threads
    firing at once just turn into 429s, burn the requests-per-minute budget on retries,
    and make the run slower than pacing it would have been. Spacing every request by
    `60 / (tpm / tokens_per_call)` keeps the pool underneath the cap instead.

    Configured by VLM_TOKENS_PER_MINUTE (0 disables it; Gemini does not need it) and
    VLM_TOKENS_PER_CALL, which is only an estimate of a scoring call's token cost. When
    the estimate is low the static pacing still draws 429s, so every 429 stretches the
    interval by 50% and a run of clean responses relaxes it back toward the configured
    value -- the pacing converges on whatever the endpoint actually allows.

    The config is read on first use, not in __init__: `bench_run.py` imports the
    evaluators (and therefore this module) before it calls `load_dotenv()`, so reading
    the environment at import time saw none of the .env settings and left the limiter
    silently disabled -- pacing only ever worked when the variables were exported by
    hand in the shell.
    """

    RETRY_BACKOFF = 1.5      # interval multiplier applied when 429s persist
    PENALTY_DEBOUNCE = 1.0   # widen at most once per this many current intervals
    RECOVER_AFTER = 8        # clean responses before easing the interval back down
    RECOVER_FACTOR = 0.8     # how much of the penalty one recovery step gives back
    MAX_INTERVAL = 30.0      # never space requests further apart than this

    def __init__(self):
        self._lock = threading.Lock()
        self._next_at = 0.0
        self._configured = False
        self._base_interval = 0.0
        self._ok_streak = 0
        self._penalized_until = 0.0
        self.interval = 0.0

    def _configure_locked(self):
        tpm = float(os.environ.get("VLM_TOKENS_PER_MINUTE", "0") or 0)
        per_call = float(os.environ.get("VLM_TOKENS_PER_CALL", "1500") or 1500)
        self._base_interval = (60.0 * per_call / tpm) if tpm > 0 else 0.0
        self.interval = self._base_interval
        self._configured = True
        if self.interval > 0:
            print(f"VLM rate limit: {tpm:.0f} TPM / {per_call:.0f} tokens per call "
                  f"-> {60.0 / self.interval:.1f} calls/min ({self.interval:.2f}s apart)")

    def wait(self):
        with self._lock:
            if not self._configured:
                self._configure_locked()
            if self.interval <= 0:
                return
            now = time.monotonic()
            start = max(now, self._next_at)
            self._next_at = start + self.interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)

    def penalize(self):
        """
        A 429 means the current spacing is still too tight; widen it.

        Debounced, because the worker pool has 32 threads in flight: one moment of going
        over the cap comes back as a burst of 429s, and widening once per response would
        raise the interval by 1.5^32 and pin it at MAX_INTERVAL over a single incident.
        Only the first 429 of each interval-long window counts, so the pacing tracks how
        long the pressure lasts rather than how many threads happened to be in flight.
        """
        with self._lock:
            if not self._configured:
                self._configure_locked()
            if self._base_interval <= 0:
                return
            self._ok_streak = 0
            now = time.monotonic()
            if now < self._penalized_until:
                return
            self._penalized_until = now + max(self.interval * self.PENALTY_DEBOUNCE, 1.0)
            widened = min(self.interval * self.RETRY_BACKOFF, self.MAX_INTERVAL)
            if widened > self.interval:
                self.interval = widened
                print(f"Rate limit hit; pacing widened to {60.0 / self.interval:.1f} calls/min")

    def reward(self):
        """Ease back toward the configured pace once the endpoint stops complaining."""
        with self._lock:
            if self._base_interval <= 0 or self.interval <= self._base_interval:
                return
            self._ok_streak += 1
            if self._ok_streak < self.RECOVER_AFTER:
                return
            self._ok_streak = 0
            self.interval = max(self._base_interval, self.interval * self.RECOVER_FACTOR)


_rate_limiter = _RateLimiter()


def score_transcript(transcript, api_pkg, temp=0.0, top_p=0.2, seed=123, max_retry=10):
    """
    Send one VLM scoring transcript and return the integer score it reports.

    `api_pkg` is the (model_type, api_key, base_url) triple the evaluators already
    carry. Returns None when every retry failed to produce a parsable score, so the
    caller can drop the sample instead of averaging in a fabricated zero.
    """
    model_type, api_key, base_url = api_pkg
    temp_now = temp
    while max_retry > 0:
        try:
            response = gptv_query(
                transcript,
                top_p=top_p,
                temp=temp_now,
                model_type=model_type,
                api_key=api_key,
                base_url=base_url,
                seed=seed,
            )
            matches = SCORE_PATTERN.findall(response)
            if matches:
                return int(matches[0][1])
        except Exception as e:
            print(f"VLM scoring call failed: {e}, retrying...")
        temp_now = min(temp_now + 0.1, 1.0)
        max_retry -= 1
    return None


def repeat_score(transcript, api_pkg, repeats=5, temp=1.0, top_p=1.0, base_seed=123, max_retry=10):
    """
    Score the same transcript `repeats` times and return every sample.

    Each repeat uses its own seed so the samples are independent draws; the spread
    across them is what the std in the reported metrics measures. Samples that never
    parsed are dropped.

    Note the default top_p of 1.0. The single-shot path uses 0.2 to pin the judge to
    its most likely answer, but truncating the distribution that hard also makes five
    repeats return the same number every time, which would report a std of 0 no matter
    how borderline the shot is. Sampling from the full distribution is what makes the
    spread mean something.
    """
    samples = []
    for i in range(max(1, int(repeats))):
        score = score_transcript(
            transcript, api_pkg, temp=temp, top_p=top_p,
            seed=(None if base_seed is None else base_seed + i), max_retry=max_retry
        )
        if score is not None:
            samples.append(score)
    return samples


def summarize_scores(samples):
    """Pool raw VLM samples into {mean, std, n}. std is the population std over the samples."""
    vals = [float(v) for v in samples if v is not None]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {"mean": mean, "std": var ** 0.5, "n": len(vals)}
