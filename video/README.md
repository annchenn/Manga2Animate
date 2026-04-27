# Comic2Animate Pipeline

A fully automated storyboard-to-animation pipeline that generates animated video clips from storyboard panels, evaluates multiple candidates using Gemini, and concatenates the best clips into a final animated story.

## Features

* Generate animation clips from storyboard images using Wan I2V
* Automatically extract related characters from the script JSON
* Generate two candidate videos for each panel
* Use Gemini to evaluate and select the better animation
* Maintain temporal coherence between consecutive clips
* Auto-resume interrupted runs
* Automatically concatenate all selected clips into a final video

---

## Pipeline Overview

```text
Storyboard Panel Image
        ↓
Upload image to ImgBB
        ↓
Generate 2 candidate videos using Wan
        ↓
Evaluate candidates with Gemini
        ↓
Select best candidate
        ↓
Repeat for all panels
        ↓
Concatenate all selected clips
        ↓
Final animated story
```

---

## Requirements

* Python 3.10+
* ffmpeg installed and added to PATH

Python packages:

```bash
pip install requests dashscope google-genai
```

---

## Project Structure

```text
project/
│
├── pipeline.py
├── best.json
├── panel1.png
├── panel2.png
├── panel3.png
├── ...
│
└── pipeline_generations/
    └── run_xxx/
        ├── panel_1/
        ├── panel_2/
        ├── ...
        └── final_story.mp4
```

---

## Input Files

### 1. Storyboard Images

Each storyboard panel should follow this naming format:

```text
panel1.png
panel2.png
panel3.png
...
```

### 2. Script JSON

The pipeline reads a full script JSON file:

```text
best.json
```

Example structure:

```json
{
  "characters": [
    {
      "id": 1,
      "name": "Character A",
      "description": "..."
    }
  ],
  "clips": [
    {
      "id": 1,
      "description": "Character[1] walks into the room."
    }
  ]
}
```

The system automatically extracts only the characters referenced in each clip.

---

## Configuration

Edit the following variables in `pipeline.py`:

```python
DASHSCOPE_API_KEY = ""
GEMINI_API_KEY = ""
IMGBB_API_KEY = ""
```

Optional settings:

```python
MODEL_NAME = "wan2.7-i2v"
VIDEO_DURATION = 3
INPUT_JSON_PATH = "best.json"
BASE_OUTPUT_DIR = "pipeline_generations"
PANEL_COUNT = 6
AUTO_RESUME = True
```

---

## How It Works

### 1. Filter Script Data

For each panel:

* Find the corresponding clip
* Extract referenced characters using:

```python
Character[ID]
```

* Build a filtered script for generation

---

### 2. Generate Videos

The pipeline:

* uploads the panel image to ImgBB
* sends generation requests to Wan
* creates two candidate videos

Each candidate is generated independently for diversity.

---

### 3. Gemini Evaluation

Gemini compares:

* Video realism
* Physical commonsense
* Prompt alignment
* Temporal coherence (for non-first clips)

The system selects:

```text
A_BETTER
B_BETTER
or
COMPARABLE
```

---

### 4. Final Concatenation

After all panels are processed:

```bash
ffmpeg concat
```

is used to merge all selected clips into:

```text
final_story.mp4
```

---

## Running the Pipeline

```bash
python pipeline.py
```

---

## Auto Resume

If execution is interrupted:

```python
AUTO_RESUME = True
```

The system automatically resumes from the latest unfinished run.

---

## Output Structure

Example:

```text
pipeline_generations/
└── run_20260427_120000/
    ├── panel_1/
    │   ├── candidate_1.mp4
    │   ├── candidate_2.mp4
    │   ├── best_panel_1.mp4
    │   ├── filtered_script.json
    │   ├── wan_prompt.txt
    │   └── gemini_eval_response.txt
    │
    ├── panel_2/
    └── ...
    │
    └── final_story.mp4
```

---

## Notes

* ffmpeg must be installed before running
* Gemini evaluation can take time because videos are uploaded and analyzed
* Wan generation is asynchronous and may require waiting for polling
* The system assumes all panel images already exist locally

---

## Evaluation Criteria

Gemini evaluates videos using:

* Visual realism
* Physical commonsense
* Prompt alignment
* Temporal coherence between clips

The evaluation is intentionally strict to improve animation quality selection.

---

## Future Improvements

* Multi-candidate tournament selection
* Better motion consistency constraints
* Automatic storyboard parsing
* Character appearance embedding consistency
* Parallel generation optimization
