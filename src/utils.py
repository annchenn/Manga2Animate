import os
import re
import time
from pathlib import Path

from google import genai

from src import metrics
from src.log import error, status, warn

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def make_genai_client() -> genai.Client:
    """Shared Gemini client factory, used by every module that calls Gemini
    (character description, script, storyboard, and reviewer stages).

    Because this is the single factory for all of them, the returned client is
    wrapped by src.metrics so every API call is counted in one place instead of
    needing per-module instrumentation.

    Prefers Vertex AI (GOOGLE_CLOUD_PROJECT/GOOGLE_CLOUD_LOCATION) when a GCP
    project is configured. Falls back to a plain API key
    (GEMINI_API_KEY) when GOOGLE_CLOUD_PROJECT is unset/empty — that path needs
    nothing more than a key from https://aistudio.google.com/apikey, no GCP IAM/
    service-account/billing setup, and stays available for anyone without GCP
    project access.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if project:
        return metrics.instrument_client(genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        ))
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        return metrics.instrument_client(genai.Client(api_key=api_key))
    raise RuntimeError(
        "No Gemini credentials found: set GOOGLE_CLOUD_PROJECT (Vertex AI) or "
        "GEMINI_API_KEY (https://aistudio.google.com/apikey) in .env."
    )


def resolve_hf_token(token_file: str, gated_repo_hint: str) -> str | None:
    """Shared HF token resolution, used by every cross-env bridge that downloads a
    gated HF model in its worker subprocess (currently
    opensource_backends/flux2/bridge.py). Priority: HF_TOKEN env var, then the given
    token file (each bridge resolves its own <MODEL>_HF_TOKEN_FILE env var/default
    before calling this). Parameterized rather than hard-coded to one model so any
    future bridge can reuse it instead of duplicating the resolution order.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    p = Path(token_file)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    warn(f"No HF token found (checked HF_TOKEN env var and {p}) — {gated_repo_hint}")
    return None


def _referenced_characters(description: str) -> list[int]:
    """Fallback extraction of Character[N] ids from a clip's description text,
    used when a script's `characters_in_panel` field is missing/empty."""
    return [int(m) for m in dict.fromkeys(re.findall(r"Character\[(\d+)\]", description))]


def characters_in_panel(clip: dict) -> list[int]:
    """Used by opensource_backends/flux2/bridge.py: prefer the script's own
    `characters_in_panel` field (added to both the flux2-exclusive and shared
    create_script prompts) when present, falling back to regexing Character[N] out of
    `description` for older scripts that predate that field."""
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


def strip_dialogue(text: str) -> str:
    """Originally _strip_dialogue inside the flux2 bridge — same logic, made public
    here so any text-conditioned image backend can reuse it. See the regex comments
    above for the bug history each pattern was written against."""
    cleaned = _SPEAKER_DIALOGUE_RE.sub("", text)
    cleaned = _DIALOGUE_RE.sub("", cleaned)
    cleaned = _ONSCREEN_TEXT_RE.sub("", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def find_character_ref_images(directory: Path) -> list[tuple[int, Path]]:
    """Character reference images for one target directory.

    New convention: <directory>/characters/char{id}.{png,jpg,jpeg}, one image per
    character, returned sorted by id. Falls back to the legacy single
    <directory>/ref.* file (treated as id 1) when characters/ doesn't exist or is
    empty, so targets that only have ref.* keep
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
    metrics.collector().record_logical()
    for attempt in range(1, max_retries + 1):
        try:
            if label:
                with status(f"{label} (attempt {attempt}/{max_retries})"):
                    return fn()
            return fn()
        except Exception as e:
            if attempt == max_retries:
                # Note this returns `default` rather than raising, so the caller
                # cannot tell the call failed -- hence recording it here.
                metrics.collector().record_failure()
                error(f"All {max_retries} attempts failed: {e}. Skipping.")
                return default
            metrics.collector().record_retry()
            warn(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)
