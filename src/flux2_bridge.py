"""
Cross-env bridge for the flux2 storyboard backend.

Runs entirely inside Manga2Animate's own conda env (pure Python — no torch/diffusers
imports here). Builds a job-spec JSON describing every panel×variant FLUX.2-dev
should generate, then shells out via `conda run -n <FLUX2_CONDA_ENV> python
src/flux2_worker.py --job <job.json>` — the ONLY file allowed to import torch/
diffusers is src/flux2_worker.py, which runs inside that separate env. Mirrors the
Skyfall-GS/difix_bridge.py + difix_worker.py pattern used elsewhere in this
workspace for the same reason (two frameworks with incompatible dependency stacks).
"""
import json
import os
import re
import subprocess
from pathlib import Path

from src.log import info, warn
from src.utils import find_character_ref_images

BASE_DIR = Path(__file__).resolve().parent.parent
WORKER_PATH = BASE_DIR / "src" / "flux2_worker.py"

FLUX2_CONDA_ENV = os.environ.get("FLUX2_CONDA_ENV", "sb_diffusers")
FLUX2_MODEL_ID = os.environ.get("FLUX2_MODEL_ID", "black-forest-labs/FLUX.2-dev")
FLUX2_NUM_VARIANTS = int(os.environ.get("FLUX2_NUM_VARIANTS", "3"))
FLUX2_PANEL_WIDTH = int(os.environ.get("FLUX2_PANEL_WIDTH", "768"))
FLUX2_PANEL_HEIGHT = int(os.environ.get("FLUX2_PANEL_HEIGHT", "640"))
FLUX2_NUM_INFERENCE_STEPS = int(os.environ.get("FLUX2_NUM_INFERENCE_STEPS", "28"))
FLUX2_GUIDANCE_SCALE = float(os.environ.get("FLUX2_GUIDANCE_SCALE", "4.0"))
FLUX2_CPU_OFFLOAD = os.environ.get("FLUX2_CPU_OFFLOAD", "true").strip().lower() not in {"false", "0", "no"}
FLUX2_WORKER_TIMEOUT = int(os.environ.get("FLUX2_WORKER_TIMEOUT", "10800"))
FLUX2_HF_TOKEN_FILE = os.environ.get(
    "FLUX2_HF_TOKEN_FILE", str(Path.home() / ".cache" / "huggingface" / "token")
)


def _referenced_characters(description: str) -> list[int]:
    """Fallback extraction of Character[N] ids from a clip's description text,
    used when a script's `characters_in_panel` field is missing/empty."""
    return [int(m) for m in dict.fromkeys(re.findall(r"Character\[(\d+)\]", description))]


def characters_in_panel(clip: dict) -> list[int]:
    ids = clip.get("characters_in_panel")
    if isinstance(ids, list) and ids:
        return [int(i) for i in ids]
    return _referenced_characters(clip.get("description", ""))


# create_script.py's prompts instruct Gemini to embed spoken dialogue and thoughts
# straight into the clip's "description" text, usually (but not reliably) prefixed
# "Dialogue:"/"Thoughts:", always in single quotes (this is the only place the
# script schema uses quotes, so matching quoted spans is a safe, convention-scoped
# heuristic here — not a general NLP problem). storyboard_test's validated recipe
# never hit this because its fixed test targets had dialogue hand-stripped out of
# every clip description once, up front (clean_script.py/targets.py) — Manga2Animate
# generates descriptions live per run, so that stripping has to happen here instead.
# FLUX.2 renders literal quoted text far more readily than Gemini's image model
# does, and the source manga panel it's editing already has a real speech bubble
# in the same spot — together those two signals reliably beat the prompt's "no
# speech bubbles" instruction unless the quoted dialogue is removed first.
# Non-greedy `.*?` + a lookahead requiring the closing quote to be followed by
# whitespace/punctuation/end-of-string (and the opening quote to NOT be preceded
# by a word character) — a naive `'[^']*'` breaks the moment the quoted dialogue
# itself contains an apostrophe (e.g. "'I'm up here!'" would only match "'I'" and
# leave "m up here!'" behind), which is common in real dialogue.
#
# Second bug found after a real run (goblin panels 4/5): `(?<!\w)` alone isn't
# enough to reject a possessive apostrophe like "Character[2]'s face..." — the
# preceding char is `]`, not a word char, so it still passes as a valid *opening*
# quote, and the non-greedy `.*?` then happily extends all the way to the *real*
# dialogue's closing quote later in the same description, deleting every bit of
# pose/composition text in between (confirmed against bestScript.json: "Character
# [2]'s face fills the lower right of the screen... wide-open mouth. Character[2]
# says: 'TO THE LIKES OF YOU!!'" collapsed to just "Character[2]"). Fixed with a
# negative lookahead right after the opening quote that rejects the apostrophe if
# it's actually a contraction/possessive suffix ('s/'t/'d/'m/'re/'ll/'ve).
#
# Third bug found on DualJustice1 (multi-line dialogue lists): Gemini sometimes
# writes "Character[1] says: 'A', 'B', and 'C'." — a cue word ("says"/"Dialogue"/
# "Thoughts") followed by a comma/"and"-joined LIST of quotes, not just one. The
# original regex only ate cue+one-quote, leaving the connectors and a second
# "says:"-less quote behind as debris ("Character[1] says: , , and ."). A bare
# "says:" immediately followed by nothing is arguably *worse* than the original
# dialogue — it reads as "there should be text here" with no content to ground
# it, which plausibly nudges FLUX.2 toward drawing an (empty) speech bubble on
# its own. Fixed two ways: (a) "says" is now itself a recognized cue word (not
# just "Dialogue:"/"Thoughts:"), so it gets removed with the quote instead of
# left dangling; (b) the pattern now consumes an entire ", "/", and "/" and "
# -joined chain of quotes as one match, not just the first.
_DIALOGUE_RE = re.compile(
    r"(?:(?:Dialogue|Thoughts?|\bsays)\s*:\s*)?"
    r"(?<!\w)'(?!(?:s|t|d|m|re|ll|ve)\b).*?'(?=[\s.,!?]|$)"
    r"(?:\s*(?:,\s*(?:and\s+)?|and\s+)(?<!\w)'(?!(?:s|t|d|m|re|ll|ve)\b).*?'(?=[\s.,!?]|$))*"
)

# Fourth bug found on a full DualJustice1 run (iter1, mode="one" — the feedback-
# driven create_script_with_one_feedback.txt template, which produces noticeably
# different phrasing than iter0's without-feedback prompt): dialogue written as a
# screenplay-style speaker-turn chain with NO quotes at all, e.g. "Dialogue:
# Character[2]: Character[3]: Character[3]: Character[2]: Character[2]:" — every
# turn is just a bare "Character[id]:" label, sometimes with a quoted line after
# it, sometimes (as here) with nothing. _DIALOGUE_RE requires a `'` to even start
# matching, so this format sailed straight through untouched and reached FLUX.2
# as literal dangling text — worse than the debris bug #3 already fixed, since
# "Dialogue: Character[2]: Character[3]: ..." reads as an explicit instruction
# that dialogue is happening here, right next to a source panel image that
# already has a real speech bubble in the same spot; the actual iter1 render
# echoed legible Japanese dialogue text into panels 1/2 as a result. Strip the
# whole cue+speaker-chain (each turn optionally followed by a quoted line, using
# the same possessive-safe quote pattern as above) before the quote-only pattern
# ever runs.
_SPEAKER_DIALOGUE_RE = re.compile(
    r"(?:Dialogue|Thoughts?)\s*:\s*"
    r"(?:Character\[\d+\]\s*:\s*(?:(?<!\w)'(?!(?:s|t|d|m|re|ll|ve)\b).*?'(?=[\s.,!?]|$)\s*)?)+"
)

# Same run, panel 3: the environment text itself described "large sound effect
# katakana surrounding the center" as a background element to draw — not a
# dialogue quote, so no dialogue-stripping pattern above ever touches it; the
# render came back with a full katakana SFX burst. This is describing on-screen
# lettering as literal visual content, so strip the naming words for it wherever
# they appear in environment/description, the same way dialogue quotes are
# stripped — leaves the surrounding scene description (speed lines, etc.) intact.
_ONSCREEN_TEXT_RE = re.compile(
    r"\b(?:\d+\s+)?(?:large |small |bold |thick |handwritten |jagged |prominent |stylized |graphical )*"
    r"(?:sound[- ]effects?|onomatopoeia|katakana(?:\s+text)?|kanji(?:\s+text)?|"
    r"speech[- ]bubbles?|dialogue[- ]boxe?s?|text[- ]bubbles?|"
    r"on[- ]screen text|written text|SFX(?:\s+text)?|lettering|"
    # Found on a full DualJustice1 run: the prompt-level instruction bans the
    # literal words above, but Gemini just paraphrased around them — "large,
    # prominent graphical letter-like strokes" describes the exact same SFX
    # burst without ever saying "katakana"/"sound effect"/"lettering". Keyword
    # blocking is inherently a losing game against paraphrase (this is the same
    # dynamic noted in build_prompt()'s docstring, just on the input side now);
    # catch the paraphrase patterns seen so far, but treat this list as never
    # complete — the Rule 9/10 prompt instruction is the first line of defense.
    r"letter[- ]like|writing[- ]like|script[- ]like|glyph[- ]like|glyphs?)\b",
    re.IGNORECASE,
)


def _strip_dialogue(text: str) -> str:
    cleaned = _SPEAKER_DIALOGUE_RE.sub("", text)
    cleaned = _DIALOGUE_RE.sub("", cleaned)
    cleaned = _ONSCREEN_TEXT_RE.sub("", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


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
    """Short form (just the summary clause) when an image reference already
    carries hair/face/outfit color detail for this character; fuller form (every
    semicolon-segment joined into one clause) when this character has NO
    reference image at all in this target, since the text is then the *only*
    identity signal FLUX.2 gets for them — char_identity_map() used to always
    truncate to the summary segment, leaving e.g. "A young man in his 20s" as the
    entire description for an uncropped character with no hair/eye/outfit color
    specified anywhere in the prompt."""
    segments = [s.strip() for s in full_desc.split(";") if s.strip()]
    if not segments:
        return "a character"
    return segments[0] if has_ref_image else "; ".join(segments)


def build_prompt(
    clip: dict, referenced_ids: list[int], char_ids_in_panel: list[int], char_identity: dict[int, str]
) -> str:
    """Ported from storyboard_test/experiments/dualjustice1_v2/
    run_flux2_perpanel_multi.py::build_prompt — validated across 5 targets — plus a few
    small additions found necessary for Manga2Animate's live-generated (not hand-cleaned)
    scripts: dialogue stripping, a no-invented-character clause, and a brief reference-image
    caveat. Keep the closing constraint sentence SHORT and mentioned ONCE: Flux2Pipeline has
    no negative_prompt, so every word here is positive conditioning to the text encoder, not
    a true negation — piling on repeated/elaborated "no text/no UI/no bubble" phrasing was
    tried and measurably made things worse (see flux2_bridge.py commit history / session
    notes), presumably because repeating a concept's tokens raises its salience regardless of
    the surrounding negation grammar. char_identity now holds each character's FULL
    description (see char_identity_map()) — _identity_phrase() below trims it to just the
    summary clause when an image reference is available, or spells out every segment
    (hair/face/outfit/etc) when there is none, since text is then the only identity signal."""
    identity_sentence = " ".join(
        f"Character[{i}] is {_identity_phrase(char_identity[i], i in char_ids_in_panel)}."
        for i in referenced_ids if i in char_identity
    )
    if char_ids_in_panel:
        ref_sentence = (
            f"The extra reference image(s) show the exact color design for Character "
            f"{' and '.join(str(i) for i in char_ids_in_panel)} — match them precisely, "
            f"ignoring any leftover text or markings in the reference image itself."
        )
    else:
        ref_sentence = ""
    # Found via a real run on goblin panel 3 (no characters_in_panel): with no
    # character described, FLUX.2 sometimes invents a generic anime character to
    # fill the frame instead of keeping the panel's actual (character-less)
    # subject — e.g. a pure sword/energy close-up became "hero holding sword in
    # energy burst". Explicit no-invention instruction when the clip has no
    # characters at all.
    no_invention_sentence = (
        "" if referenced_ids else
        "This panel has no character in it according to the script — do not add any person "
        "or character that is not already present in the original panel image."
    )
    environment = _strip_dialogue(clip.get("environment", ""))
    description = _strip_dialogue(clip.get("description", ""))
    return (
        f"Redraw this manga panel as a single fully-colored anime storyboard frame. "
        f"{identity_sentence} "
        f"{environment} {description} "
        f"{ref_sentence} "
        f"{no_invention_sentence} "
        f"Fully colorize the entire frame in a clean, finished anime illustration style with flat "
        f"cel shading; keep hairstyle, eye color, and outfit colors identical to the reference "
        f"image(s) where provided. The frame must be a clean full-color illustration only — no "
        f"black-and-white line art, screentone, speech bubbles, dialogue boxes, or on-screen text "
        f"of any kind. Preserve the exact camera angle, framing, rotation, and character pose "
        f"shown in the input panel image — do not straighten, re-center, or flatten a dramatic or "
        f"angled shot into a standard front-facing composition."
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
        "cpu_offload": FLUX2_CPU_OFFLOAD,
        "num_variants": FLUX2_NUM_VARIANTS,
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "panels": panels,
    }


def _resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    p = Path(FLUX2_HF_TOKEN_FILE)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    warn(
        f"No HF token found (checked HF_TOKEN env var and {p}) — FLUX.2-dev is a gated "
        f"HuggingFace repo, so the worker subprocess will likely fail to download it. Either "
        f"set HF_TOKEN directly, run `huggingface-cli login`, or point FLUX2_HF_TOKEN_FILE at "
        f"a token file. The account behind the token must have accepted the license at "
        f"https://huggingface.co/black-forest-labs/FLUX.2-dev"
    )
    return None


def run_worker(job_spec_path: Path) -> None:
    env = os.environ.copy()
    token = _resolve_hf_token()
    if token:
        env["HF_TOKEN"] = token
    hf_home = os.environ.get("FLUX2_HF_HOME")
    if hf_home:
        env["HF_HOME"] = hf_home

    cmd = [
        "conda", "run", "-n", FLUX2_CONDA_ENV, "--no-capture-output",
        "python", str(WORKER_PATH), "--job", str(job_spec_path),
    ]
    info(f"Launching flux2 worker: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, timeout=FLUX2_WORKER_TIMEOUT)
    except FileNotFoundError as e:
        raise RuntimeError(
            "`conda` executable not found on PATH. Ensure conda's bin directory is on "
            "PATH, then retry."
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

    missing = [str(f) for files in expected.values() for f in files if not f.exists()]
    if missing:
        raise RuntimeError(
            f"flux2 worker finished but {len(missing)} candidate image(s) are still "
            f"missing (first few: {missing[:5]})"
        )
    return expected
