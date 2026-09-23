# MangaEval

MangaEval scores manga-to-animation results at two levels: the **storyboard**
(6 panels) and the **video**. It builds on ViStoryBench and VBench++, with the
modifications described in the paper (Sec. 4.1.3). You can use it to evaluate
any method's output, as long as you provide the source manga material for each
test case (see [Inputs](#inputs)).

Metric names follow the paper (Sec. 4.1.3).

| level | category | metric (short name) | code |
|---|---|---|---|
| storyboard | Visual Consistency (V.C.) | CSD | `vistorybench/` (`csd`) |
| storyboard | Visual Consistency (V.C.) | CIDS, **SigLIP-384 instead of CLIP-224** | `vistorybench/` (`cids`) |
| storyboard | Content Alignment (C.A.) | PA, **GPT-4.1 judge, 5 repeats averaged** | `vistorybench/` (`prompt_align`) |
| storyboard | Content Alignment (C.A.) | OCCM, **reference count detected on the source manga panel** | `vistorybench/` (reported with `cids`) |
| storyboard | Aesthetic Quality (A.Q.) | Aesthetic Shadow v2 (Shadow-v2), Waifu Scorer v3 (Waifu-v3) | `aesthetic/` |
| storyboard | Computational Cost | VLM calls, generation calls, time | counted from the run outputs, no script |
| video | Manga Alignment | I2V Subject Similarity (I2VS), **DINOv3 ViT-B/16 backbone** | `vbench/tools/evaluate_i2v_subject_dinov3_official.py` |
| video | Manga Alignment | I2V Background (I2VB) | `vbench/evaluate_i2v.py` |
| video | Video Quality (V.Q.) | Subject Consistency (SC), Background Consistency (BC), Motion Smoothness (MS) | `vbench/evaluate_i2v.py` |
| video | Video Quality (V.Q.) | Dynamic Degree (DD), **continuous mean flow magnitude, × 256/min(H,W)** | `vbench/tools/modified_dynamic_degree.py` |
| video | Aesthetic Quality (A.Q.) | Aesthetic Shadow v2 (Shadow-v2), Waifu Scorer v3 (Waifu-v3), 8 frames sampled evenly per video, averaged | `aesthetic/` |
| video | Computational Cost | VLM calls, generation calls, time | counted from the run outputs, no script |

Bold text marks where MangaEval departs from the original benchmark.

```
mangaeval/
├── crop_video_frames.py   # video → per-shot frames, so storyboard metrics can score a video
├── vistorybench/          # storyboard metrics (ViStoryBench + our changes)
├── vbench/                # video metrics (VBench++ + our two modified metrics in tools/)
└── aesthetic/             # anime aesthetic scorers
```

## Inputs

Each test case is one folder under `vistorybench/data/dataset/ViStory/<story_id>/`,
where `<story_id>` is a two-digit id (`01`, `02`, ...):

```
vistorybench/data/dataset/ViStory/
├── story_index.json             # {"<case name>": "<story_id>", ...}
└── 01/
    ├── story.json               # characters + one entry per shot (format below)
    ├── image/<character>/00.jpg # one reference image per character (jpg; run_bench.sh converts)
    └── manga_panels/panel1.png  # the source manga panels, panel1..panel6, for OCCM
```

`story.json` follows the ViStoryBench format:

```json
{
  "Story_type": {"en": "Fantasy Adventure Manga"},
  "Characters": {
    "elf1": {"name_en": "elf1", "tag": "unrealistic_human",
             "prompt_en": "Young adult female elf with long silver-white hair ..."}
  },
  "Shots": [
    {"index": 1,
     "Plot Correspondence": {"en": "..."},
     "Setting Description": {"en": "..."},
     "Shot Perspective Design": {"en": "..."},
     "Characters Appearing": {"en": ["elf1"]},
     "Static Shot Description": {"en": "..."}}
  ]
}
```

### Example case

`vistorybench/data/dataset/ViStory/01/` is a complete test case, ready to score,
taken from [Manga109](http://www.manga109.org/) (*Dual Justice*, page 18): its
`story.json`, three character references and the six manga panels. To try the
pipeline end to end, put a method's six panels at
`vistorybench/data/outputs/<method>/base/en/<timestamp>/01/shots/shot_1.png ... shot_6.png`
and run `./run_bench.sh --method <method> --mode base`.

`vistorybench/src/convert_to_vistory.py` builds these folders from a
Manga2Animate run (`bestScript.json` → `story.json`, using Gemini to fill the
fields), and `vistorybench/src/link_manga_panels.py` copies the cropped manga
panels in. Without `manga_panels/`, OCCM falls back to the character count in
`story.json` and prints a warning.

## Storyboard metrics (`vistorybench/`)

### Setup

```bash
cd mangaeval/vistorybench
conda create -n vistorybench python=3.11 -y
conda activate vistorybench
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install "mkl=2023.1.0" "numpy=1.26.4"   # the line above pulls versions that break torch 2.4
pip install -r requirements.txt
sh download_weights.sh                          # model weights into data/pretrain/ (~9GB)
git clone https://github.com/mk-minchul/AdaFace vistorybench/bench/content/AdaFace
```

PA calls a VLM judge. Put the judge settings in `vistorybench/.env`:

```
BASE_URL=https://api.openai.com/v1
MODEL_ID=gpt-4.1
API_KEY=sk-...
VLM_TOKENS_PER_MINUTE=30000   # client-side rate limit; 0 disables it
VLM_TOKENS_PER_CALL=1500
```

Use the same judge for every method in one table: scores from different judges
are not comparable. If `API_KEY` is empty the evaluator falls back to
`GEMINI_API_KEY` without a warning, so set it explicitly.

### Outputs to score

Place each method's 6 panels at

```
data/outputs/<method>/base/en/<YYYYMMDD_HHMMSS>/<story_id>/shots/shot_1.png ... shot_6.png
```

File names must end in the shot number (`shot_1.png` works, `panel1.png` crashes
the loader) and follow the shot order in `story.json`.

### Run

```bash
./run_bench.sh --method <method> --mode base                              # cids, csd, prompt_align
./run_bench.sh --method <method> --mode base --metrics cids csd           # local only, no API cost
GPU=1 ./run_bench.sh --method <method> --mode base
```

`run_bench.sh` converts reference images to jpg, runs `bench_run.py`, and writes
a readable `result.md` next to the results in `data/bench_results/`. PA settings
(5 repeats, temperature 1.0) and the SigLIP backbone for CIDS are in
`config.yaml`.

## Video metrics

### Storyboard metrics on a video (`crop_video_frames.py`)

To score a video with the storyboard metrics, extract frames per shot first.
For a video that contains all 6 shots back to back (the Manga2Animate
reference-conditioned mode), `crop_video_frames.py` detects the 5 cuts from
HSV-histogram distance and samples 10 frames inside each shot:

```bash
# the newest `create_video.py -m seedance` run: results/<case>/video/seedance/run_*/best_video.mp4
python mangaeval/crop_video_frames.py -t <case name> -m seedance --method <method>_video

# any other video
python mangaeval/crop_video_frames.py -t <case name> --video path/to/video.mp4 --method <method>_video
```

`-m` is the model folder under `results/<case>/video/`. The newest run that has a
final video (`best_video.mp4`, or `final_story.mp4` for the first-frame mode) is
used. The frames land in
`vistorybench/data/outputs/<method>_video/base/en/<timestamp>/<story_id>/shots/shot_<n>/`,
with a copy and a contact sheet in `video_frames/` next to the video.
Every frame is scored and averaged per shot; CIDS self-similarity uses only the
middle frame of each shot. Run from the repo root.

### VBench++ metrics (`vbench/`)

All video metrics are computed per shot: each video is split into one clip per
panel and the scores are averaged over the clips. The I2V metrics compare each
clip with a reference image: I2VS with the character reference, I2VB with the
source manga panel.

```bash
cd mangaeval/vbench
pip install -r requirements.txt && pip install -e .

# split every video in ./need_clip/ into its 6 panels:
#   need_clip/<video name>/panel_01.mp4 ... panel_06.mp4  (+ transitions.txt)
mkdir -p need_clip && cp <videos>/*.mp4 need_clip/
python auto_clip.py

# I2VB: --custom_image_folder holds the manga panel for each clip
python evaluate_i2v.py --mode custom_input --ratio 16-9 --dimension i2v_background \
    --videos_path need_clip/<video name> --custom_image_folder <manga_panels> --output_path out/

# SC, BC, MS
python evaluate_i2v.py --mode custom_input --ratio 16-9 \
    --dimension subject_consistency background_consistency motion_smoothness \
    --videos_path need_clip/<video name> --output_path out/
```

`auto_clip.py` finds the 5 panel transitions from global frame differences,
keeps the best-scoring combination of candidates at least 0.5 s apart, and
refines each boundary locally. Detected boundaries are written to
`transitions.txt`. It skips videos that already have all six clips; set
`FORCE_RECUT = True` at the top of the script to redo them.

`--custom_image_folder` is paired with the videos **by file name**: the
reference for `panel_01.mp4` must be named `panel_01.png` (or `.jpg`), and so
on. Every video in `--videos_path` needs a matching image, otherwise the run
stops with a `KeyError`. Use `_` rather than `-` before the number: VBench
drops a trailing `-<number>` from the name, so `panel-01.mp4` would be matched
to `panel.png`.

The two modified metrics read a JSON manifest, a list of one entry per clip
(`segment` is the panel number).
`method` and `case` are free-form names; `skip` is optional:

```json
[
  {"case": "01", "segment": 1, "method": "mymethod",
   "video_path": "need_clip/myvideo/panel_01.mp4", "character_ref": "/path/to/girl1.png"},
  {"case": "01", "segment": 2, "method": "mymethod",
   "video_path": "need_clip/myvideo/panel_02.mp4", "character_ref": "/path/to/girl1.png",
   "skip": "NO_CHARACTER"}
]
```

For I2VS, a clip with `"skip"` set is recorded with status
`SKIP_<reason>` and left out of the average instead of being scored. Use it
when no character appears in the clip (`NO_CHARACTER`) or the reference
image does not show the character (`INVALID_REFERENCE`); these are the two
reasons we used in the paper. DD ignores `skip` and
`character_ref`.

```bash
# I2VS with DINOv3 ViT-B/16 (same scoring formula as VBench++)
pip install -r requirements-dinov3.txt
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3_official
git -C third_party/dinov3_official checkout 6876159a11b4df116f30f667f8c9888617df0751
python tools/evaluate_i2v_subject_dinov3_official.py --manifest manifest.json \
    --output out/i2v_subject.csv --method mymethod --gpu 0

# DD, continuous
python tools/modified_dynamic_degree.py --manifest manifest.json \
    --output out/dynamic_degree.csv --method mymethod \
    --checkpoint ~/.cache/vbench/raft_model/models/raft-things.pth
```

Weights:

- **DINOv3 ViT-B/16**: gated. Request access at
  https://github.com/facebookresearch/dinov3 (download form in its README), then
  put `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` in
  `third_party/dinov3_official/`. The script checks its hash.
- **RAFT** (`raft-things.pth`, for DD): VBench downloads it to
  `~/.cache/vbench/raft_model/models/` (or under `$VBENCH_CACHE_DIR`) the first
  time `evaluate_i2v.py --dimension dynamic_degree` runs. To fetch it directly:

  ```bash
  wget -P ~/.cache/vbench/raft_model/ https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
  unzip -d ~/.cache/vbench/raft_model/ ~/.cache/vbench/raft_model/models.zip
  ```

- The other VBench++ checkpoints (e.g. AMT for MS, DINO for SC) are downloaded automatically on first use.

## Aesthetic quality (`aesthetic/`)

Aesthetic Shadow v2 (Shadow-v2; ViT classifier, P(high quality) in 0-1) and
Waifu Scorer v3 (Waifu-v3; CLIP ViT-L/14 + MLP head, 0-10) are no-reference scores, so they apply to
any image or video. The tool has its own conda env:

```bash
bash mangaeval/aesthetic/setup_env.sh          # creates env manga2animate-aesthetic
conda activate manga2animate-aesthetic

# storyboard: the 6 panels of each method
python mangaeval/aesthetic/score.py <panels_dir> --out storyboard_aesthetic.csv

# video: 8 frames sampled evenly from each video (the default; no cut detection)
python mangaeval/aesthetic/score.py <videos_dir> --out video_aesthetic.csv
```

The CSV has one row per image (or video frame) and metric; the terminal prints
the mean per file. The reported value is the mean over a method's 6 panels
(storyboard) or over the 8 sampled frames (video). The first run downloads the weights (~4GB for Shadow-v2 into the
Hugging Face cache, ~1GB CLIP backbone into `~/.cache/clip`). Shadow-v2's
weights are CC-BY-NC-4.0 (non-commercial). Details: `aesthetic/README.md`.
