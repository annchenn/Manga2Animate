# Comic2Animate Video Pipeline

A storyboard-to-animation pipeline that generates animated video clips from manually prepared storyboard panels, evaluates multiple candidates using Gemini, and concatenates the best clips into a final animated story.

This README is for the `Manga2Animate/video/` folder.

---

## Features

* Generate animation clips from storyboard panel images using Wan I2V.
* Use manually prepared storyboard panels as the first frame for video generation.
* Calculate dynamic video duration based on the original manga panel size and dialogue word count.
* Automatically extract related characters from the script JSON.
* Generate two candidate videos for each panel.
* Use Gemini to evaluate and select the better animation.
* Maintain temporal coherence between consecutive clips.
* Optional auto-resume for interrupted runs.
* Concatenate all selected clips into a final video using FFmpeg.

---

## Requirements

* Python 3.10+
* FFmpeg installed and available in your system PATH.

Python packages:

```bash
pip install requests dashscope google-genai Pillow
```

---

## Folder Structure

Expected structure from the `Manga2Animate/` project root:

```text
Manga2Animate/
├── results/
│   └── {target}/
│       ├── iter0/
│       │   ├── bestScript.json
│       │   └── bestStoryboard.png
│       └── ...
│
├── target/
│   └── {target}/
│       ├── manga.png           
│       ├── ref.png
│       ├── manga_panels/
│       │   ├── panel1.png
│       │   ├── panel2.png
│       │   └── ...
│       └── storyboard_panels/
│           ├── panel1.png
│           ├── panel2.png
│           └── ...
│
└── video/
    ├── pipeline.py
    ├── README.md
    └── pipeline_generations/
```

---

## Input Files

The pipeline uses a target argument, such as `-t death_notebook`.
The target name should match folders under both `results/` and `target/`.

### 1. Script JSON

The pipeline automatically loads the latest iteration script from:
`Manga2Animate/results/{target}/iter_max/bestScript.json`

### 2. Full Manga Page (For Ratio Calculation)

To calculate how long a panel should be shown, the pipeline compares the panel's size to the full page. You must provide the original full manga page:
`Manga2Animate/target/{target}/manga.png`

*(Note: If your full page has a different filename, you can specify it using the `--manga-page` argument).*

### 3. Storyboard Panels (For Video Generation)

The first-frame images for Wan I2V should be placed manually under:
`Manga2Animate/target/{target}/storyboard_panels/panel{i}.png`

### 4. Original Manga Panels (For Duration Estimation)

The original manga panel crops should be placed under:
`Manga2Animate/target/{target}/manga_panels/panel{i}.png`

These manga panels are compared against `manga.png` to calculate the exact area ratio.

---

### 3. Original Manga Panels for Duration Estimation

The original manga panel crops should be placed under:

```text
Manga2Animate/target/{target}/manga_panels/
```

Example:

```text
Manga2Animate/target/death_notebook/manga_panels/panel1.png
Manga2Animate/target/death_notebook/manga_panels/panel2.png
Manga2Animate/target/death_notebook/manga_panels/panel3.png
Manga2Animate/target/death_notebook/manga_panels/panel4.png
Manga2Animate/target/death_notebook/manga_panels/panel5.png
Manga2Animate/target/death_notebook/manga_panels/panel6.png
```

These manga panels are used only for dynamic duration calculation.

---

## Configuration

Edit the following API keys directly in `pipeline.py`:

```python
DASHSCOPE_API_KEY = "YOUR_DASHSCOPE_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
IMGBB_API_KEY = "YOUR_IMGBB_KEY"
```

Main settings:

```python
MODEL_NAME = "wan2.7-i2v"
PANEL_COUNT = 6
```

Input and output paths are built automatically from the target argument.

---

## How It Works

### 1. Load Input Data

For each target, the pipeline loads:

```text
results/{target}/iter_max/bestScript.json
target/{target}/storyboard_panels/panel{i}.png
target/{target}/manga_panels/panel{i}.png
```

Example:

```text
results/death_notebook/iter1/bestScript.json
target/death_notebook/storyboard_panels/panel1.png
target/death_notebook/manga_panels/panel1.png
```

---

### 2. Filter Script Data and Calculate Duration

For each panel:

* Finds the corresponding clip in `bestScript.json`.
* Extracts referenced characters, such as `Character[1]`.
* Calculates optimal duration between 2 and 5 seconds based on:
  * original manga panel area
  * dialogue word count

---

### 3. Generate Candidate Videos

For each storyboard panel:

* Uploads `storyboard_panels/panel{i}.png` to ImgBB.
* Sends two independent generation requests to Wan I2V.
* Downloads the results as:

```text
candidate_1.mp4
candidate_2.mp4
```

---

### 4. Gemini Evaluation

The pipeline copies the candidates as:

```text
video_A.mp4
video_B.mp4
```

Gemini evaluates them based on:

* Visual Realism
* Physical Commonsense
* Video-Prompt Alignment
* Coherency with the previous selected clip

Gemini returns:

```text
A_BETTER
B_BETTER
COMPARABLE
```

The selected video is saved as:

```text
best_panel_{i}.mp4
```

---

### 5. Final Concatenation

After all panels are processed, the pipeline creates:

```text
concat_list.txt
```

Then FFmpeg concatenates all selected clips into:

```text
final_story.mp4
```

---

## Running the Pipeline

From the project root:

```bash
python video/pipeline.py -t death_notebook
```

From inside the `video/` folder:

```bash
python pipeline.py -t death_notebook
```

Resume from the latest interrupted run:

```bash
python video/pipeline.py -t death_notebook --auto-resume
```

Set the number of panels:

```bash
python video/pipeline.py -t death_notebook --panel-count 6
```

If your full manga page has a different filename
```bash
python video/pipeline.py -t death_notebook --manga-page my_custom_page.jpg
```

---

## Output Structure

Generated outputs are saved under:

```text
Manga2Animate/video/pipeline_generations/{target}/
```

Example:

```text
Manga2Animate/video/pipeline_generations/death_notebook/
└── run_20260504_120000/
    ├── panel_1/
    │   ├── input.png
    │   ├── candidate_1.mp4
    │   ├── candidate_2.mp4
    │   ├── best_panel_1.mp4
    │   ├── filtered_script.json
    │   ├── wan_prompt.txt
    │   ├── gemini_eval_prompt.txt
    │   ├── gemini_eval_response.txt
    │   └── tasks_info.json
    │
    ├── panel_2/
    ├── panel_3/
    ├── panel_4/
    ├── panel_5/
    ├── panel_6/
    ├── concat_list.txt
    └── final_story.mp4
```

---

## Notes

* `video/pipeline_generations/` should usually be added to `.gitignore`.
* `storyboard_panels/panel{i}.png` is used as the first frame for Wan I2V.
* `manga_panels/panel{i}.png` is used only for duration estimation.
* `bestScript.json` is loaded from the latest `iterX` folder under `results/{target}/`.
* FFmpeg is required for the final concatenation step.
* Gemini evaluation may take time because videos must be uploaded and processed.
* Wan generation is asynchronous; the script polls Dashscope every 15 seconds.
