"""
Cross-env bridge for the flux2 storyboard backend.

Runs entirely inside Manga2Animate's own conda env (pure Python — no torch/diffusers
imports here). Builds a job-spec JSON describing every panel×variant FLUX.2-dev
should generate, then shells out via `conda run -n <FLUX2_CONDA_ENV> python
opensource_backends/flux2/worker.py --job <job.json>` — the ONLY file allowed to
import torch/diffusers is worker.py, which runs inside that separate env (this
repo's orchestrator env and FLUX.2's own diffusers/torch stack have incompatible
dependency pins, so they cannot share one env).

This module also owns GPU selection. That is not cosmetic: the one local (non-RunPod)
run of this backend died because another process held ~40GB of the chosen card, so
FLUX.2 OOMed on the second panel and every subsequent call then failed with a
confusing device-mismatch cascade (enable_model_cpu_offload leaves its hooks
half-moved after an OOM). Pinning the worker to a specific idle GPU via
CUDA_VISIBLE_DEVICES, plus a pre-flight free-VRAM check in worker.py, turns that
class of failure into an explicit error *before* any weights are loaded.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from src import metrics
from src.log import info, warn
from src.utils import (
    characters_in_panel,
    find_character_ref_images,
    resolve_hf_token,
    strip_dialogue as _strip_dialogue,
)

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
WORKER_PATH = BACKEND_DIR / "worker.py"

FLUX2_CONDA_ENV = os.environ.get("FLUX2_CONDA_ENV", "sb_diffusers")
FLUX2_MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-dev")
FLUX2_NUM_VARIANTS = int(os.environ.get("FLUX2_NUM_VARIANTS", "3"))
FLUX2_PANEL_WIDTH = int(os.environ.get("FLUX2_PANEL_WIDTH", "768"))
FLUX2_PANEL_HEIGHT = int(os.environ.get("FLUX2_PANEL_HEIGHT", "640"))
FLUX2_NUM_INFERENCE_STEPS = int(os.environ.get("FLUX2_NUM_INFERENCE_STEPS", "28"))
FLUX2_GUIDANCE_SCALE = float(os.environ.get("FLUX2_GUIDANCE_SCALE", "4.0"))
# "resident" (default) loads the text encoder, encodes every panel prompt once,
# frees it, then keeps transformer+vae (~64.8GB) resident on one card for the whole
# panel x variant loop — no per-call weight streaming at all. "cpu_offload" is the
# older, slower path (keeps ~113GB in CPU RAM and streams submodules to the GPU on
# EVERY pipe() call); it is kept as a fallback for cards too small for resident mode.
FLUX2_OFFLOAD_MODE = os.environ.get("FLUX2_OFFLOAD_MODE", "resident").strip().lower()
# Legacy knob: FLUX2_CPU_OFFLOAD=false used to mean `pipe.to("cuda")`, which cannot
# work for FLUX.2-dev (113GB of bf16 weights vs. a 97GB card — confirmed OOM), so it
# now only selects between the two modes above.
if "FLUX2_OFFLOAD_MODE" not in os.environ and "FLUX2_CPU_OFFLOAD" in os.environ:
    FLUX2_OFFLOAD_MODE = (
        "cpu_offload"
        if os.environ["FLUX2_CPU_OFFLOAD"].strip().lower() not in {"false", "0", "no"}
        else "resident"
    )
# Default true: from_pretrained() otherwise hits huggingface.co to re-validate
# every file's metadata on EVERY load, even when the local snapshot is already 100%
# complete — found via a real run that hung 7+ minutes at "Fetching 31 files: 0%"
# with a socket open to the Hub's API, not reading local disk at all. Since this
# worker only ever reads a cache whose completeness verify_setup.sh already
# confirmed, there is nothing for that network round-trip to accomplish here. Set
# to false only for a first-time download onto a FRESH/incomplete cache (that one
# run should also raise FLUX2_WORKER_TIMEOUT, since it now has to fetch ~105GB).
FLUX2_HF_OFFLINE = os.environ.get("FLUX2_HF_OFFLINE", "true").strip().lower() not in {"false", "0", "no"}
# Variants of one panel share a prompt, so they CAN be generated in a single pipe()
# call (num_images_per_prompt) with one generator per variant. Measured on this
# machine (RTX PRO 6000 Blackwell, 768x640, 28 steps): batch=1 was 82.0s/image,
# batch=3 was 82.7s/image -- i.e. no wall-clock benefit at all, because the GPU is
# already compute-saturated per image; batching just spreads the same linear cost
# across one call instead of three. It DOES cost more peak VRAM for no payoff
# (63.6GB at batch=1 vs 72.0GB at batch=3), which matters more than usual here
# since this is a shared machine where another process can appear on a GPU that
# looked idle moments earlier. Default 1 for that reason; raise it only if a
# future GPU/resolution combination turns out not to be compute-saturated.
# Capped by FLUX2_NUM_VARIANTS at use site; worker.py falls back to 1 on OOM anyway.
FLUX2_BATCH_VARIANTS = int(os.environ.get("FLUX2_BATCH_VARIANTS", "1"))
# "auto" picks the GPU with the least memory in use (see pick_cuda_device).
FLUX2_CUDA_DEVICE = os.environ.get("FLUX2_CUDA_DEVICE", "auto").strip()
FLUX2_MIN_FREE_VRAM_GB = float(os.environ.get("FLUX2_MIN_FREE_VRAM_GB", "0") or 0)
FLUX2_MAX_CONSECUTIVE_FAILURES = int(os.environ.get("FLUX2_MAX_CONSECUTIVE_FAILURES", "2"))
FLUX2_WORKER_TIMEOUT = int(os.environ.get("FLUX2_WORKER_TIMEOUT", "10800"))
FLUX2_HF_TOKEN_FILE = os.environ.get(
    "FLUX2_HF_TOKEN_FILE", str(Path.home() / ".cache" / "huggingface" / "token")
)


# characters_in_panel() / _referenced_characters() / strip_dialogue() now live in
# src/utils.py, since they are generic script-schema helpers rather than anything
# FLUX.2-specific — imported above as _strip_dialogue for this module's own call
# sites below.


def char_identity_map(character_description: dict, best_script: dict | None = None) -> dict[int, str]:
    """{'characters': [{'id':1,'description':'A;B;C'}]} -> {1: 'A;B;C'} — full
    semicolon-segmented description per id (build_prompt() decides how much of
    it to actually use, depending on whether an image reference is available).

    best_script's own 'characters' array — when given — overrides the entry from
    character_description.json for any id it covers. create_script...txt's Rules
    1 & 8 explicitly tell Gemini to silently correct a character's description
    against the actual manga visuals into this array when it conflicts with
    character_description.json, but character_description.json itself is never
    rewritten with that correction — so reading only character_description.json
    here would keep serving the pre-correction text forever. Confirmed as a real
    bug via a DualJustice1 run where character_description.json described id 2 as
    a 'humanoid mecha' while the actual characters/char2.png was a dragon and
    best_script's corrected entry matched the manga; without this override,
    build_prompt() fed FLUX.2 the wrong identity sentence for that character on
    every panel it appeared in."""
    out: dict[int, str] = {}
    for c in character_description.get("characters", []):
        desc = c.get("description", "").strip()
        if desc:
            out[int(c["id"])] = desc
    for c in (best_script or {}).get("characters", []):
        desc = c.get("description", "").strip()
        if desc:
            out[int(c["id"])] = desc  # best_script's corrected entry wins
    return out


def _identity_phrase(full_desc: str, has_ref_image: bool) -> str:
    """Always the fuller form (every semicolon-segment joined into one clause),
    regardless of whether this character has a reference image in this panel.

    Used to truncate to just the summary segment (e.g. "Athletic woman") when
    has_ref_image was True, on the assumption that the reference image itself
    would carry hair/face/outfit color detail — confirmed WRONG via a real
    DualJustice1 run: Character[3]'s image reference was attached on every panel
    she appeared in, but her design (hairstyle, tunic color, cape) still visibly
    drifted panel-to-panel, landing closest to the reference only on panel 5 (the
    one panel whose pose happened to closely match the reference image's own
    pose). The reference image alone isn't reliable enough to carry full identity
    on its own — pairing it with the complete text description (hair color, outfit
    colors, etc.) gives FLUX.2 two independent identity signals instead of one,
    which is also what Black Forest Labs' own prompting guide recommends (describe
    the subject explicitly rather than relying on an image reference to do all the
    work). has_ref_image is kept as a parameter for now (harmless / self-documenting
    at call sites) even though this function no longer branches on it."""
    segments = [s.strip() for s in full_desc.split(";") if s.strip()]
    if not segments:
        return "a character"
    return "; ".join(segments)


def build_prompt(
    clip: dict, referenced_ids: list[int], char_ids_in_panel: list[int], char_identity: dict[int, str]
) -> str:
    """Ported from storyboard_test/experiments/dualjustice1_v2/
    run_flux2_perpanel_multi.py::build_prompt — validated across 5 targets — plus a few
    small additions found necessary for Manga2Animate's live-generated (not hand-cleaned)
    scripts: dialogue stripping, a no-invented-character clause, and a brief reference-image
    caveat. char_identity now holds each character's FULL description (see
    char_identity_map()) — _identity_phrase() below trims it to just the summary clause
    when an image reference is available, or spells out every segment (hair/face/outfit/
    etc) when there is none, since text is then the only identity signal.

    Three changes made against Black Forest Labs' official FLUX.2 prompting guide
    (docs.bfl.ml/guides/prompting_guide_flux2), targeted at specific documented
    limitations rather than a wholesale rewrite — the rest of this prompt
    (natural-language, no JSON; lighting carried by the script's own environment
    field) already matches that guide:
    1. "Word order matters — FLUX.2 pays more attention to what comes first." The
       camera-angle/pose-preservation constraint used to be the LAST sentence in the
       whole prompt; it's the guide's own recommended #3 priority (critical
       style/constraint, right after subject+action), not an afterthought, so it's
       stated right after identity_sentence now instead of at the very end.
    2. "For multi-reference editing, describe how each input should be used... subject
       from image 1, style from image 2." With 2+ referenced characters in one panel,
       the old ref_sentence was one blanket line covering all reference images at once,
       never saying which image was which character's — a plausible source of two
       characters' color designs bleeding into each other. Now describes each reference
       image's role individually, numbered to match Flux2Pipeline's actual
       images=[panel_img, *ref_imgs] order in flux2_worker.py (panel is image 1, so
       reference images start at 2).
    3. "FLUX.2 does not support negative prompts. Focus on describing what you want,
       not what you don't want." composition_sentence, the empty-characters branch of
       no_invention_sentence, and the closing colorize sentence used to end in a "do
       not X" / "no X, no Y, no Z" clause each; all three are now phrased as the
       desired end state instead (e.g. "no black-and-white line art, screentone,
       speech bubbles..." became "rendered purely as finished color artwork, exactly
       as a professional anime production still"). This is a deliberate reversal of an
       earlier empirical finding noted here previously — that piling on MORE negative
       phrasing measurably made things worse — on the theory that the earlier result
       was about *over-elaborating* the negation, not about negation being inherently
       fine in moderation; going by the guide's own worked example ("no blurry
       background" -> "sharp focus throughout") instead. NOT yet re-validated against
       the model (no fresh generation run since this change) — if a future run shows
       the specific failure modes these sentences guard against (B&W line art,
       screentone, speech bubbles, invented characters, flattened composition)
       creeping back in, that is the first thing to check."""
    identity_sentence = " ".join(
        f"Character[{i}] is {_identity_phrase(char_identity[i], i in char_ids_in_panel).rstrip('.')}."
        for i in referenced_ids if i in char_identity
    )
    # Positive-framed per Black Forest Labs' FLUX.2 prompting guide ("FLUX.2 does not
    # support negative prompts. Focus on describing what you want, not what you don't
    # want.") — states the desired end state (angle kept exactly as dramatic as the
    # source) instead of the forbidden one (don't straighten/flatten/re-center).
    composition_sentence = (
        "Preserve the exact camera angle, framing, rotation, and character pose shown in the "
        "input panel image, keeping any dramatic or angled shot exactly as dramatic and angled "
        "as the source panel."
    )
    if len(char_ids_in_panel) == 1:
        ref_sentence = (
            f"The reference image shows the exact color design for Character "
            f"{char_ids_in_panel[0]} — match it precisely, ignoring any leftover text or "
            f"markings in the reference image itself."
        )
    elif char_ids_in_panel:
        ref_sentence = " ".join(
            f"Reference image {idx} shows the exact color design for Character {cid} — match "
            f"it precisely, ignoring any leftover text or markings in the reference image itself."
            for idx, cid in enumerate(char_ids_in_panel, start=2)
        )
    else:
        ref_sentence = ""
    # Found via a real run on goblin panel 3 (no characters_in_panel): with no
    # character described, FLUX.2 sometimes invents a generic anime character to
    # fill the frame instead of keeping the panel's actual (character-less)
    # subject — e.g. a pure sword/energy close-up became "hero holding sword in
    # energy burst". Explicit no-invention instruction when the clip has no
    # characters at all.
    # Positive-framed for the same reason as composition_sentence: states what the
    # frame DOES contain (the panel's existing non-character subject matter, exactly
    # as shown) rather than what must not be added.
    no_invention_sentence = (
        "" if referenced_ids else
        "This panel has no character in it according to the script, so the frame contains "
        "only the original panel's existing subject matter — its objects, effects, and "
        "environment — exactly as shown, with zero people present."
    )
    environment = _strip_dialogue(clip.get("environment", ""))
    description = _strip_dialogue(clip.get("description", ""))
    return (
        f"Redraw this manga panel as a single fully-colored anime storyboard frame. "
        f"{identity_sentence} "
        f"{composition_sentence} "
        f"{environment} {description} "
        f"{ref_sentence} "
        f"{no_invention_sentence} "
        f"Fully colorize the entire frame in a clean, finished anime illustration style with flat "
        f"cel shading, covering every surface in smooth, solid color; keep hairstyle, eye color, "
        f"and outfit colors identical to the reference image(s) where provided. The frame shows "
        f"only the drawn scene itself — characters, objects, and background — rendered purely as "
        f"finished color artwork, exactly as a professional anime production still."
    )


def build_job_spec(
    best_script: dict,
    character_description: dict,
    target_dir: Path,
    panels_dir: Path,
    out_dir: Path,
    manifest_path: Path,
) -> dict:
    ref_images = dict(find_character_ref_images(target_dir))
    char_identity = char_identity_map(character_description, best_script)

    panels = []
    for clip in sorted(best_script["clips"], key=lambda c: c["id"]):
        cid = clip["id"]
        panel_image = panels_dir / f"panel{cid}.png"
        if not panel_image.exists():
            raise FileNotFoundError(f"Source manga panel missing: {panel_image}")

        referenced = characters_in_panel(clip)
        crop_ids = [i for i in referenced if i in ref_images]
        for i in referenced:
            if i not in char_identity and i not in ref_images:
                warn(
                    f"panel {cid}: Character[{i}] has neither a text identity "
                    f"(character_description.json) nor a reference crop (characters/char{i}.*) "
                    f"— flux2 has no identity signal for this character in this panel."
                )

        panels.append(
            {
                "id": cid,
                "prompt": build_prompt(clip, referenced, crop_ids, char_identity),
                "panel_image": str(panel_image),
                "reference_images": [str(ref_images[i]) for i in crop_ids],
                "crop_ids": crop_ids,
            }
        )

    return {
        "model_id": FLUX2_MODEL_ID,
        "panel_width": FLUX2_PANEL_WIDTH,
        "panel_height": FLUX2_PANEL_HEIGHT,
        "num_inference_steps": FLUX2_NUM_INFERENCE_STEPS,
        "guidance_scale": FLUX2_GUIDANCE_SCALE,
        "offload_mode": FLUX2_OFFLOAD_MODE,
        "batch_variants": max(1, min(FLUX2_BATCH_VARIANTS, FLUX2_NUM_VARIANTS)),
        "min_free_vram_gb": FLUX2_MIN_FREE_VRAM_GB,
        "max_consecutive_failures": FLUX2_MAX_CONSECUTIVE_FAILURES,
        "num_variants": FLUX2_NUM_VARIANTS,
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "panels": panels,
    }


def _resolve_hf_token() -> str | None:
    return resolve_hf_token(
        FLUX2_HF_TOKEN_FILE,
        "FLUX.2-dev is a gated HuggingFace repo, so the worker subprocess will likely fail to "
        "download it. Either set HF_TOKEN directly, run `huggingface-cli login`, or point "
        "FLUX2_HF_TOKEN_FILE at a token file. The account behind the token must have accepted "
        "the license at https://huggingface.co/black-forest-labs/FLUX.2-dev",
    )


def _resolve_conda_exe() -> str:
    """Absolute path to a usable `conda`, or the bare name as a last resort.

    `conda` is frequently NOT on PATH in a non-interactive shell even when conda
    envs are in active use (conda's shell init only runs for login/interactive
    shells), so assuming the bare name makes the backend fail with a bare
    FileNotFoundError on machines where conda is perfectly healthy. Resolution
    order: explicit override, conda's own CONDA_EXE, PATH, then inferred from this
    interpreter's prefix -- an env at <root>/envs/<name> implies <root>/bin/conda.
    """
    override = os.environ.get("FLUX2_CONDA_EXE")
    if override:
        return override
    for candidate in (os.environ.get("CONDA_EXE"), shutil.which("conda")):
        if candidate and Path(candidate).exists():
            return candidate
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        inferred = prefix.parent.parent / "bin" / "conda"
        if inferred.exists():
            return str(inferred)
    for base in (Path.home() / "miniconda3", Path.home() / "anaconda3"):
        if (base / "bin" / "conda").exists():
            return str(base / "bin" / "conda")
    return "conda"


def pick_cuda_device() -> str | None:
    """Which GPU index to expose to the worker, as a CUDA_VISIBLE_DEVICES value.

    FLUX2_CUDA_DEVICE is used verbatim when set to anything but "auto" (so
    "0", "1", or even "1,3" all pass straight through). "auto" asks nvidia-smi for
    per-GPU used memory and picks the least-loaded card.

    Why this exists: see the module docstring — a foreign process holding memory on
    the default card is what broke the one local run of this backend, and the GPU a
    pipeline lands on is otherwise entirely implicit. Returns None if nvidia-smi is
    unavailable, in which case the worker inherits whatever the parent sees and the
    pre-flight VRAM check in worker.py is the remaining safety net.
    """
    if FLUX2_CUDA_DEVICE and FLUX2_CUDA_DEVICE.lower() != "auto":
        return FLUX2_CUDA_DEVICE
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        warn(f"Could not run nvidia-smi to auto-pick a GPU ({e}) — leaving CUDA_VISIBLE_DEVICES unset.")
        return None

    gpus = []
    for line in out.strip().splitlines():
        try:
            idx, used = (part.strip() for part in line.split(","))
            gpus.append((int(used), int(idx)))
        except ValueError:
            continue
    if not gpus:
        warn("nvidia-smi returned no parseable GPU rows — leaving CUDA_VISIBLE_DEVICES unset.")
        return None

    used_mib, idx = min(gpus)
    others = ", ".join(f"GPU{i}={u}MiB" for u, i in sorted(gpus, key=lambda g: g[1]))
    info(f"Auto-selected GPU {idx} ({used_mib}MiB already in use) — [{others}]")
    if used_mib > 1024:
        warn(
            f"GPU {idx} is the emptiest card but already holds {used_mib}MiB from another "
            f"process. FLUX.2-dev needs most of a 97GB card; set FLUX2_CUDA_DEVICE to pick "
            f"a different GPU, or wait for that process to finish."
        )
    return str(idx)


def run_worker(job_spec_path: Path) -> None:
    env = os.environ.copy()
    token = _resolve_hf_token()
    if token:
        env["HF_TOKEN"] = token
    hf_home = os.environ.get("FLUX2_HF_HOME")
    if hf_home:
        env["HF_HOME"] = hf_home
    if FLUX2_HF_OFFLINE:
        env["HF_HUB_OFFLINE"] = "1"

    device = pick_cuda_device()
    if device is not None:
        # The worker then sees exactly one card as cuda:0, which is what both offload
        # modes target. Doing it here (not in the worker) keeps the worker free of any
        # device-discovery logic and makes the choice visible in the launch log below.
        env["CUDA_VISIBLE_DEVICES"] = device
    # Recommended by the OOM message from the failed local run; lets the allocator
    # grow a segment instead of fragmenting across many fixed-size blocks.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [
        _resolve_conda_exe(), "run", "-n", FLUX2_CONDA_ENV, "--no-capture-output",
        "python", str(WORKER_PATH), "--job", str(job_spec_path),
    ]
    info(f"Launching flux2 worker: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, timeout=FLUX2_WORKER_TIMEOUT)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not execute conda ({_resolve_conda_exe()!r}). Set FLUX2_CONDA_EXE to "
            f"the absolute path of your conda binary (e.g. ~/miniconda3/bin/conda), or put "
            f"conda's bin directory on PATH, then retry."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"flux2_worker timed out after {FLUX2_WORKER_TIMEOUT}s "
            f"(FLUX2_WORKER_TIMEOUT env var) — raise it for first-run HF downloads."
        ) from e
    if result.returncode != 0:
        raise RuntimeError(
            f"flux2_worker exited with code {result.returncode} in conda env "
            f"'{FLUX2_CONDA_ENV}' — see subprocess output above."
        )


def _collect_worker_metrics(manifest_path: Path) -> None:
    """Fold the worker's own timings into this run's metrics.

    The worker runs in a different conda env, so rather than passing anything back
    over the subprocess boundary it writes a "_metrics" entry into the candidates
    manifest it already rewrites after every batch. Reading it here keeps the two
    sides decoupled and still gets GPU timings and peak VRAM into metrics.json.
    """
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        warn(f"Could not read flux2 worker metrics from {manifest_path}: {e}")
        return
    worker_metrics = manifest.get("_metrics")
    if not isinstance(worker_metrics, dict):
        return
    metrics.add("flux2", worker_metrics)
    info(
        f"flux2 worker: {worker_metrics.get('images_generated_this_run', '?')} image(s) in "
        f"{worker_metrics.get('generation_s', '?')}s "
        f"(mode={worker_metrics.get('offload_mode', '?')}, "
        f"batch={worker_metrics.get('batch_variants', '?')}, "
        f"peak VRAM {worker_metrics.get('peak_vram_reserved_gb', '?')}GB)"
    )


def generate_candidates(
    best_script: dict,
    character_description: dict,
    target_dir: Path,
    panels_dir: Path,
    storyboard_dir: Path,
) -> dict[int, list[Path]]:
    """Returns {panel_id: [candidate image paths, v1..vN]}. Resumable: skips the
    subprocess entirely if every expected file already exists (auto-resume)."""
    out_dir = storyboard_dir / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "flux2_candidates_meta.json"

    job_spec = build_job_spec(
        best_script, character_description, target_dir, panels_dir, out_dir, manifest_path
    )
    job_spec_path = storyboard_dir / "flux2_job.json"
    job_spec_path.write_text(json.dumps(job_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    expected = {
        p["id"]: [out_dir / f"panel{p['id']}_v{v}.png" for v in range(1, job_spec["num_variants"] + 1)]
        for p in job_spec["panels"]
    }
    if all(f.exists() for files in expected.values() for f in files):
        info("All flux2 candidate images already exist — skipping worker subprocess")
    else:
        run_worker(job_spec_path)

    _collect_worker_metrics(manifest_path)

    missing = [str(f) for files in expected.values() for f in files if not f.exists()]
    if missing:
        raise RuntimeError(
            f"flux2 worker finished but {len(missing)} candidate image(s) are still "
            f"missing (first few: {missing[:5]})"
        )
    return expected
