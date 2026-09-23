"""Video generation backends, and which pipeline route each one can run.

Two routes exist (see `create_video.py`):

  ref  — the whole `bestStoryboard.png` goes in as a *reference image* and the
         model returns one continuous video for the entire story. Only models
         that accept a reference image can do this.
  i2v  — the storyboard is cut into panels and each panel drives its own clip,
         which are reviewed against the previous clip and concatenated.

A backend declares `mode` and the driver picks the route from it, so adding a
model is a matter of adding one entry to `BACKENDS` here.

Interfaces a backend implements:

  submit(prompt, image, duration, seed) -> task handle   \\  async remote APIs
  poll(task_id)                         -> url or path   /

  generate_batch(jobs, out_dir, seed)   -> list[Path]       local, batched

`batch = True` backends run every panel in one process invocation instead of
one call per panel — for a local model, loading the weights once per run rather
than once per clip is the difference between minutes and hours.

`needs_public_url` says the API only takes an http(s) image, so the driver
uploads to ImgBB first; the others are handed a local `Path`.
"""

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from src import log

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Locally-hosted open-source weights live under one root, one directory per
# model, so a backend only has to name its directory. Override the root with
# OSVG_MODELS_DIR, or a single model's directory with <MODEL>_WEIGHTS.
OSVG_MODELS_DIR = Path(
    os.getenv("OSVG_MODELS_DIR", PROJECT_ROOT / "opensource_video_generator" / "models")
)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")   # wan
ARK_API_KEY       = os.getenv("ARK_API_KEY")         # seedance
MINIMAX_API_KEY   = os.getenv("MINIMAX_API_KEY")     # hosted minimax
IMGBB_API_KEY     = os.getenv("IMGBB_API_KEY")

WAN_URL           = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
WAN_TASK_URL      = "https://dashscope-intl.aliyuncs.com/api/v1/tasks"
SEEDANCE_URL      = "https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks"
MINIMAX_URL       = "https://api.minimax.io/v2/video_generation"
MINIMAX_QUERY_URL = "https://api.minimax.io/v2/query/video_generation"

# The hosted MiniMax API caps the text prompt at 7000 characters. This is an API
# limit, not a model limit: the self-hosted sglang path enforces no length at all
# (Qwen3-VL-32B's text encoder has max_position_embeddings=262144), so only the
# `minimax-api` backend checks it.
MINIMAX_PROMPT_LIMIT = 7000

# H3 is the only MiniMax model this pipeline targets, hosted or self-hosted.
# `MiniMax-H3-Max` accepts reference input too, so it could drive the ref route —
# it trades resolution and length for speed (480P/768P, no 2K, 5-15s against H3's
# 768P/2K, 4-15s). Point MINIMAX_MODEL at it to try it.
MINIMAX_DEFAULT_MODEL = "MiniMax-H3"


class PolicyViolationError(RuntimeError):
    """The provider refused the prompt/image. Retrying the same input won't help."""

    def __init__(self, task_id, message):
        super().__init__(message)
        self.task_id = task_id
        self.message = message


# ======================
# SHARED HELPERS
# ======================
def upload_image_to_imgbb(image_path):
    file_size_mb = os.path.getsize(image_path) / 1024 / 1024
    log.step(f"Uploading to ImgBB ({file_size_mb:.2f} MB)...")
    import base64
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    res = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": IMGBB_API_KEY, "image": encoded},
        timeout=60,
    )
    res.raise_for_status()
    data = res.json()
    if not data.get("success"):
        raise RuntimeError(_imgbb_error(data))
    url = data["data"]["url"]
    log.info(f"ImgBB → {url}")
    return url


# A full account is the failure this hits in practice, and ImgBB reports it as a generic
# rejection — so say so plainly instead of leaving a raw payload to decode. The symptom
# downstream is worse than the upload error itself: the video provider is handed a URL it
# cannot fetch and fails with "media url unreachable", which reads like a network problem.
_IMGBB_FULL_HINTS = ("storage", "full", "limit", "quota", "exceed", "space")


def _imgbb_error(data):
    message = str(((data or {}).get("error") or {}).get("message", "")) or str(data)
    if any(hint in message.lower() for hint in _IMGBB_FULL_HINTS):
        log.warn("ImgBB looks FULL — delete some images at https://imgbb.com/ and retry.")
        return (
            f"ImgBB upload rejected, and the message looks like a storage limit: {message}\n"
            f"The account is probably full — delete some images at https://imgbb.com/ "
            f"and run the same command again."
        )
    return f"ImgBB upload failed: {data}"


def _unreachable_image_hint(payload):
    """Point at ImgBB when a provider says it could not fetch the reference image.

    The reference image is handed over as an ImgBB URL, so "media url unreachable" is
    almost never the video provider's fault -- it is the image not being served. A full
    ImgBB account is the usual cause, and nothing in the provider's message says so.
    """
    if "unreachable" in str(payload).lower():
        log.warn("The video provider could not fetch the reference image from ImgBB. "
                 "If ImgBB is FULL, delete some images at https://imgbb.com/ and retry.")


def _as_data_uri(image):
    """Inline a local image as a base64 data URI; pass an http(s) URL through unchanged."""
    text = str(image)
    if text.startswith("http://") or text.startswith("https://") or text.startswith("data:"):
        return text
    import base64
    import mimetypes
    path = Path(text[len("file://"):] if text.startswith("file://") else text)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    log.info(f"Inlining {path.name} for MiniMax ({len(encoded) // 1024} KB base64)")
    return f"data:{mime};base64,{encoded}"


def fetch_result(result, out_path):
    """A backend's poll() returns either an http url or a local path — save either."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = str(result)
    if text.startswith("http://") or text.startswith("https://"):
        out_path.write_bytes(requests.get(text, timeout=600).content)
    else:
        src = text[len("file://"):] if text.startswith("file://") else text
        shutil.copy(src, out_path)
    return out_path


# ======================
# BACKEND BASE
# ======================
@dataclass
class Backend:
    name: str
    mode: str                       # "ref" | "i2v"
    model_id: str
    ref_duration: int = 15          # length of the single full-storyboard video
    panel_clamp: tuple = (2, 5)     # (min, max) seconds for one i2v panel clip
    needs_public_url: bool = False
    batch: bool = False
    is_local: bool = False
    env_keys: tuple = ()            # env vars that must be set before running
    weights_dirname: str = ""       # local models: directory under OSVG_MODELS_DIR
    prompt_limit: int = 0           # max prompt characters, 0 = no limit

    @property
    def weights_dir(self):
        if not self.weights_dirname:
            return None
        override = os.getenv(f"{self.name.upper().replace('-', '_')}_WEIGHTS")
        return Path(override) if override else OSVG_MODELS_DIR / self.weights_dirname

    def check_env(self):
        missing = [k for k in self.env_keys if not os.getenv(k)]
        if missing:
            raise RuntimeError(
                f"{self.name}: missing environment variable(s): {', '.join(missing)}"
            )
        if self.needs_public_url and not IMGBB_API_KEY:
            raise RuntimeError(f"{self.name}: IMGBB_API_KEY is required to upload the reference image")

    def build_panel_prompt(self, clip, characters=None):
        """Per-panel text for the i2v route. None = use the pipeline's
        `video_generation.txt` template with the clip JSON, which is what every
        API model gets. A backend with a small text budget overrides this."""
        return None

    def submit(self, prompt, image, duration, seed=None):
        """`seed` is honoured only where the previous scripts used one (anisora).
        The hosted providers deliberately ignore it: their two candidates are two
        identical requests told apart by sampling, which is how every comparison
        run so far was produced."""
        raise NotImplementedError

    def poll(self, task_id, interval=15, timeout=1800):
        raise NotImplementedError

    def generate_batch(self, jobs, out_dir, seed=None):
        raise NotImplementedError


# ======================
# WAN  (DashScope, i2v)
# ======================
class WanBackend(Backend):
    def submit(self, prompt, image, duration, seed=None):
        headers = {
            "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        payload = {
            "model": self.model_id,
            "input": {
                "prompt": prompt,
                "media": [{"type": "first_frame", "url": image}],
            },
            "parameters": {
                "size": "1K",
                "duration": duration,
                "prompt_extend": False,
                "watermark": False,
            },
        }
        res = requests.post(WAN_URL, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["output"]["task_id"]

    def poll(self, task_id, interval=15, timeout=1800):
        headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = requests.get(f"{WAN_TASK_URL}/{task_id}", headers=headers).json()
            status = res["output"]["task_status"]
            log.info(f"Task ...{task_id[-8:]} → {status}")
            if status == "SUCCEEDED":
                return res["output"]["video_url"]
            if status in ("FAILED", "CANCELED"):
                _unreachable_image_hint(res)
                raise RuntimeError(f"Task {task_id} failed: {res}")
            time.sleep(interval)
        raise TimeoutError(f"Polling timed out after {timeout}s")


# ======================
# SEEDANCE  (BytePlus Ark, reference image)
# ======================
class SeedanceBackend(Backend):
    def submit(self, prompt, image, duration, seed=None):
        headers = {
            "Authorization": f"Bearer {ARK_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image}, "role": "reference_image"},
            ],
            "ratio": "16:9",
            "resolution": "720p",
            "duration": duration,
            "generate_audio": True,
        }
        res = requests.post(SEEDANCE_URL, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["id"]

    def poll(self, task_id, interval=15, timeout=1800):
        headers = {"Authorization": f"Bearer {ARK_API_KEY}"}
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = requests.get(f"{SEEDANCE_URL}/{task_id}", headers=headers)
            res.raise_for_status()
            data = res.json()
            status = data.get("status")
            log.info(f"Task ...{task_id[-8:]} → {status}")
            if status == "succeeded":
                return data["content"]["video_url"]
            if status in ("failed", "cancelled"):
                error = data.get("error", {}) or {}
                if "SensitiveContent" in error.get("code", "") or "PolicyViolation" in error.get("code", ""):
                    raise PolicyViolationError(task_id, error.get("message", "Policy violation"))
                _unreachable_image_hint(data)
                raise RuntimeError(f"Task {task_id} failed: {data}")
            if status == "expired":
                raise RuntimeError(f"Task {task_id} expired.")
            time.sleep(interval)
        raise TimeoutError(f"Polling timed out after {timeout}s")


# ======================
# MINIMAX — hosted api.minimax.io
# ======================
class MiniMaxApiBackend(Backend):
    """Hosted MiniMax. The reference image travels inline, not as a URL.

    MiniMax's fetcher could not reach our ImgBB links -- every task failed with
    `content[1].image_url: media url unreachable` while the same URL returned HTTP 200
    from this machine and Seedance consumed it without trouble. A base64 data URI takes
    the fetch out of the loop entirely, which is what the API accepts here.
    """

    def submit(self, prompt, image, duration, seed=None):
        headers = {
            "Authorization": f"Bearer {MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_id,
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _as_data_uri(image)},
                 "role": "reference_image"},
            ],
            "duration": duration,
            "resolution": "768P",
            "ratio": "16:9",
        }
        res = requests.post(MINIMAX_URL, headers=headers, json=payload)
        if res.status_code >= 400:
            # The most likely 4xx here is a model id this account cannot use, which
            # the raw HTTP error does not make obvious.
            raise RuntimeError(
                f"MiniMax rejected the request ({res.status_code}) for model "
                f"'{self.model_id}': {res.text[:300]}\n"
                f"If that id is wrong or your account lacks access to it, set "
                f"MINIMAX_MODEL in .env to the id shown in the platform console."
            )
        data = res.json()
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise RuntimeError(f"MiniMax submission returned no task id: {data}")
        return task_id

    def poll(self, task_id, interval=15, timeout=1800):
        headers = {"Authorization": f"Bearer {MINIMAX_API_KEY}"}
        deadline = time.time() + timeout
        while time.time() < deadline:
            res = requests.get(f"{MINIMAX_QUERY_URL}/{task_id}", headers=headers)
            res.raise_for_status()
            # v2 nests everything under "task": {id, status, error, content: {url}, ...}
            task = res.json().get("task", {})
            status = (task.get("status") or "").lower()
            log.info(f"Task ...{task_id[-8:]} → {status}")
            if status == "succeeded":
                return task["content"]["url"]
            if status in ("failed", "cancelled"):
                error = task.get("error", {}) or {}
                blob = f"{error.get('code', '')} {error.get('message', '')}".lower()
                if "sensitive" in blob or "policy" in blob or "violat" in blob:
                    raise PolicyViolationError(task_id, error.get("message", "Policy violation"))
                _unreachable_image_hint(task)
                raise RuntimeError(f"Task {task_id} failed: {task}")
            time.sleep(interval)
        raise TimeoutError(f"Polling timed out after {timeout}s")


# ======================
# MINIMAX — self-hosted MiniMax-H3 behind sglang
# ======================
class MiniMaxLocalBackend(Backend):
    """`opensource_backends/minimax_h3/serve.sh` must already be running.

    The request shape comes from sglang's own `minimax_h3/request_validation.py`,
    not the cookbook: the generic OpenAI `seconds`/`size` fields are ignored by
    this model, and `VideoGenerationsRequest` is `extra="allow"`, so wrong fields
    are accepted at the door and fail deep inside the H3 stage instead.
    """

    port: int = int(os.getenv("MINIMAX_LOCAL_PORT", "30011"))
    steps: int = int(os.getenv("MINIMAX_LOCAL_STEPS", "50"))

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1/videos"

    def check_env(self):
        try:
            requests.get(self.base_url, timeout=5)
        except requests.RequestException as e:
            raise RuntimeError(
                f"No MiniMax-H3 server on port {self.port} ({e}). The client talks to a "
                f"running sglang server, it does not load the weights itself — start it with "
                f"`bash opensource_backends/minimax_h3/serve.sh` "
                f"(weights: {self.weights_dir}), or point MINIMAX_LOCAL_PORT at an existing one."
            ) from e

    def submit(self, prompt, image, duration, seed=None):
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "task": "ref2va",
            "conditions": [
                {"type": "image", "uri": f"file://{Path(image).resolve()}", "role": "reference"}
            ],
            # short_edge must be 768; duration_seconds must be in [4.0, 15.0].
            "target": {
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": float(duration),
            },
            "num_outputs_per_prompt": 1,
            "num_inference_steps": self.steps,
        }
        # Unlike the hosted providers, this one does take a seed: sglang declares
        # `seed` on VideoGenerationsRequest and threads it into minimax_h3's
        # canonical request. Leaving it out makes every candidate identical
        # (byte-identical files), so the A/B judge would compare a file with itself.
        if seed is not None:
            payload["seed"] = int(seed)
        res = requests.post(self.base_url, json=payload, timeout=600)
        res.raise_for_status()
        return res.json()["id"]

    def poll(self, task_id, interval=15, timeout=10800):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            r = requests.get(f"{self.base_url}/{task_id}", timeout=60)
            r.raise_for_status()
            job = r.json()
            status = job.get("status")
            if status != last:
                log.info(f"Task ...{str(task_id)[-8:]} → {status} ({job.get('progress')}%)")
                last = status
            if status in ("completed", "succeeded"):
                src = job.get("file_path") or (job.get("file_paths") or [None])[0] or job.get("url")
                if not src:
                    raise RuntimeError(f"no video in finished job: {json.dumps(job)[:500]}")
                return src
            if status in ("failed", "cancelled", "error"):
                raise RuntimeError(f"job {task_id} {status}: {job.get('error')}")
            time.sleep(interval)
        raise TimeoutError(f"job {task_id} not done within {timeout}s")


# ======================
# ANISORA  (local, batched i2v)
# ======================
#
# Prompt format is not ours to choose here: anisora's T5 text encoder tops out at
# 512 tokens, so the pipeline's `video_generation.txt` (7 rules + the whole
# bestScript.json) does not fit. Our open-source model comparison settled on
# a condensed per-panel form instead — condensed RULES, then that one clip's
# `environment` + `description` verbatim with dialogue kept, then anisora's
# recommended aesthetic/motion/no-text suffix — and that is what is reproduced
# below, wording unchanged, so runs from here are comparable with that round.
ANISORA_RULES = (
    "Rules: Anime style, natural motion. Strictly follow the script; "
    "do not add characters, objects, or backgrounds not described. "
    "Do not render any text, speech bubbles, or sound-effect text. Script: "
)
ANISORA_SUFFIX = " aesthetic score: 5.5. motion score: 3.0. There is no text in the video."


class AnisoraBackend(Backend):
    """Index-anisora V3.2 through its own conda env and CLI.

    Unlike every other backend there is no server: generation is a synchronous
    process that holds the GPU. It also takes a whole prompt_list at once, so all
    panels of one candidate run in a single invocation — the weights load once
    per candidate instead of once per panel.

    Frames, not seconds: 16fps, and the frame count must be 8k+1. It is also a
    per-invocation flag, so every panel in a batch comes out the same length.
    """

    dry_run: bool = False

    @property
    def anisora_dir(self):
        """The V3.2 subtree of the checkout — where generate_txt_new.py lives."""
        return self.weights_dir / "anisoraV3.2"

    def check_env(self):
        if not self.anisora_dir.is_dir():
            raise RuntimeError(
                f"anisora checkout not found: {self.anisora_dir}. Clone Index-anisora into "
                f"{self.weights_dir}, or point ANISORA_WEIGHTS at it (see opensource_backends/README.md)."
            )

    def build_panel_prompt(self, clip, characters=None):
        """The condensed single-line form — see the note above this class."""
        text = f"{ANISORA_RULES}{clip.get('environment', '').strip()} {clip.get('description', '').strip()}"
        return " ".join(text.split()) + ANISORA_SUFFIX

    @staticmethod
    def frames_for(duration):
        """Seconds → anisora's frame count at 16fps, rounded to the nearest 8k+1."""
        return max(9, int(round(duration * 16 / 8)) * 8 + 1)

    def generate_batch(self, jobs, out_dir, seed=None):
        """jobs: [{"prompt":..., "image": Path, "duration": int}] in panel order."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # One prompt_list line per panel: {prompt}@@{image}&&{position of the
        # conditioning frame}. 0.5 puts the panel at the middle of the clip.
        lines = [
            f"{' '.join(j['prompt'].split())}@@{Path(j['image']).resolve()}&&0.5"
            for j in jobs
        ]
        prompt_list = out_dir / "prompt_list.txt"
        prompt_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # --frame_num is per invocation, so one length covers the whole batch.
        # The default is 81 (5s); ANISORA_FRAME_NUM=auto instead takes
        # the longest clip the duration heuristic asked for.
        frame_env = os.getenv("ANISORA_FRAME_NUM", "81")
        if frame_env == "auto":
            frames = max(self.frames_for(j["duration"]) for j in jobs)
        else:
            frames = int(frame_env)

        gpus = os.getenv("ANISORA_GPU", "0,1")
        ngpu = len([g for g in gpus.split(",") if g.strip()])

        if ngpu > 1:
            # 1280*720 x 81f OOMs on a single 48GB card, so 2+ GPUs go through
            # torchrun with FSDP + ulysses sequence parallelism. --t5_cpu (not
            # --t5_fsdp) keeps ~5.5GB/GPU free for the VAE's encode transient.
            launch = [
                "torchrun", f"--nproc_per_node={ngpu}",
                f"--master_port={os.getenv('ANISORA_MASTER_PORT', '23461')}",
                "generate_txt_new.py", "--dit_fsdp", "--t5_cpu", "--ulysses_size", str(ngpu),
            ]
        else:
            # The expert checkpoints are stored fp32 (54GB each); --convert_model_dtype
            # casts to bf16 (~27GB) so one expert fits on a 48GB card.
            launch = ["python", "generate_txt_new.py", "--t5_cpu", "--convert_model_dtype"]

        cmd = [
            "conda", "run", "--no-capture-output", "-n", os.getenv("ANISORA_ENV", "anisora3.2"),
            *launch,
            "--task", "i2v-A14B",
            "--size", os.getenv("ANISORA_SIZE", "1280*720"),
            "--ckpt_dir", "Index-anisora-V3.2/V3.2",
            "--ckpt_dir_lowname", "low_noise_model",
            "--ckpt_dir_highname", "high_noise_model",
            "--prompt_list", str(prompt_list),
            "--save_dir", str(out_dir),
            "--sample_steps", os.getenv("ANISORA_STEPS", "8"),
            "--sample_shift", "5",
            "--sample_guide_scale", "1",
            "--frame_num", str(frames),
            "--base_seed", str(seed if seed is not None else 4096),
        ]

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        log.info(f"anisora: {len(jobs)} panel(s), {frames} frames @16fps, GPU {gpus}")
        if self.dry_run:
            log.warn("--dry-run: not launching anisora. Command:")
            log.info(f"cd {self.anisora_dir} && CUDA_VISIBLE_DEVICES={gpus} " + " ".join(cmd))
            return []

        subprocess.run(cmd, cwd=self.anisora_dir, env=env, check=True)
        return self.collect_outputs(jobs, out_dir)

    @staticmethod
    def output_name(job):
        """The file generate_txt_new.py will write for this job.

        From its line 428:
            save_file = save_dir/{basename(image) without extension}_{formatted_prompt}.mp4
        where formatted_prompt is the prompt with " " and "/" replaced by "_",
        truncated to 50 characters. So the name is fully determined by our own
        input, not by the order anisora processed the list in.
        """
        formatted = job["prompt"].replace(" ", "_").replace("/", "_")[:50]
        return f"{Path(job['image']).stem}_{formatted}.mp4"

    @classmethod
    def collect_outputs(cls, jobs, out_dir):
        """Map anisora's files back to panel order, by exact expected filename."""
        out_dir = Path(out_dir)
        found = []
        for job in jobs:
            path = out_dir / cls.output_name(job)
            found.append(path if path.exists() else None)
        missing = [j["index"] for j, p in zip(jobs, found) if p is None]
        if missing:
            log.warn(f"anisora produced no mp4 for panel(s) {missing} — check {out_dir}")
        return found


# ======================
# REGISTRY
# ======================
def _env(name, default):
    return os.getenv(name, default)


def build_registry():
    return {
        "wan": WanBackend(
            name="wan", mode="i2v",
            model_id=_env("WAN_MODEL", "wan2.7-i2v"),
            ref_duration=5, panel_clamp=(2, 5),
            needs_public_url=True, env_keys=("DASHSCOPE_API_KEY",),
        ),
        "seedance": SeedanceBackend(
            name="seedance", mode="ref",
            model_id=_env("SEEDANCE_MODEL", "dreamina-seedance-2-0-260128"),
            ref_duration=15, panel_clamp=(4, 5),
            needs_public_url=True, env_keys=("ARK_API_KEY",),
        ),
        "minimax-api": MiniMaxApiBackend(
            name="minimax-api", mode="ref",
            model_id=_env("MINIMAX_MODEL", MINIMAX_DEFAULT_MODEL),
            ref_duration=15, panel_clamp=(4, 6),
            # Not needs_public_url: MiniMax cannot reliably fetch our ImgBB links
            # (it fails the task with "media url unreachable" while the same URL serves
            # fine from here and to Seedance), so the image is inlined instead.
            needs_public_url=False, env_keys=("MINIMAX_API_KEY",),
            prompt_limit=MINIMAX_PROMPT_LIMIT,
        ),
        "minimax-local": MiniMaxLocalBackend(
            name="minimax-local", mode="ref",
            model_id=_env("MINIMAX_LOCAL_MODEL", "MiniMaxAI/MiniMax-H3"),
            ref_duration=15, panel_clamp=(4, 15),
            is_local=True, weights_dirname="MiniMax-H3",
        ),
        "anisora": AnisoraBackend(
            name="anisora", mode="i2v",
            model_id="Index-anisora-V3.2",
            ref_duration=5, panel_clamp=(2, 5),
            batch=True, is_local=True, weights_dirname="Index-anisora",
        ),
    }


BACKEND_NAMES = ["wan", "seedance", "minimax-local", "minimax-api", "anisora"]

def detect_backend(model_id):
    """Guess the backend from a generation model id.

    MiniMax is the one ambiguous case: the self-hosted path is addressed by a
    HuggingFace repo id (`MiniMaxAI/MiniMax-H3`), the hosted API by a bare name,
    so the "/" is what separates them.
    """
    m = (model_id or "").lower()
    if not m:
        return None
    if "wan" in m:
        return "wan"
    if "anisora" in m:
        return "anisora"
    if "minimax" in m or "hailuo" in m or m.startswith("h3"):
        return "minimax-local" if "/" in m else "minimax-api"
    if "seedance" in m or "dreamina" in m:
        return "seedance"
    return None


def model_from_env():
    """The backend to use when -m is omitted, detected from VIDEO_GENERATION_MODEL.

    Only the backend comes from it — the exact model id always comes from that
    backend's own variable (SEEDANCE_MODEL, WAN_MODEL, ...). Letting this one
    string do both jobs meant `-m seedance` and an omitted `-m` could submit
    different model ids from the same .env.
    """
    gen = os.getenv("VIDEO_GENERATION_MODEL", "").strip()
    detected = detect_backend(gen)
    if detected:
        return detected

    raise ValueError(
        f"No model given. Pass -m ({', '.join(BACKEND_NAMES)}), or set "
        f"VIDEO_GENERATION_MODEL to a model id a backend can be detected from"
        + (f" ('{gen}' matched none)" if gen else "")
    )


def get_backend(name):
    registry = build_registry()
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choices: {', '.join(BACKEND_NAMES)}")
    return registry[name]
