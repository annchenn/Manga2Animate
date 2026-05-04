---

# Comic2Animate Pipeline

A fully automated storyboard-to-animation pipeline that generates animated video clips from storyboard panels, evaluates multiple candidates using Gemini, and concatenates the best clips into a final animated story.

## Features

* Generate animation clips from storyboard images (`panel{i}.png`) using Wan I2V.
* **Intelligent Dynamic Duration (2-5 seconds):** Automatically calculates the optimal video length based on the original manga panel size (`manga_panel{i}.png`) and the word count of the dialogue.
* Automatically extract related characters from the script JSON.
* Generate two candidate videos for each panel to ensure diversity.
* Use Gemini 3.1 Pro to evaluate and select the better animation.
* Maintain temporal coherence between consecutive clips.
* Optional auto-resume for interrupted runs.
* Automatically concatenate all selected clips into a final video using FFmpeg.

---

## Requirements

* Python 3.10+
* **FFmpeg** installed and added to your system's PATH.

Python packages:

```bash
pip install requests dashscope google-genai Pillow
```

---

## Project Structure

```text
project/
│
├── pipeline.py
├── best.json
├── panel1.png              # Used as the first frame for video generation
├── manga_panel1.png        # Used to calculate dynamic duration based on area
├── panel2.png
├── manga_panel2.png
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

### 1. Generation Base Images
The cleaned or processed images used as the first frame for Wan I2V generation:
```text
panel1.png
panel2.png
...
```

### 2. Original Manga Panels (For Timing)
The pipeline needs the original manga panel crops to accurately calculate the visual importance (area) of the scene.
```text
manga_panel1.png
manga_panel2.png
...
```

### 3. Script JSON
The pipeline reads a full script JSON file (`best.json`). Example structure:

```json
{
  "characters":[
    {
      "id": 1,
      "description": "Teenage boy, slim build..."
    }
  ],
  "clips":[
    {
      "id": 1,
      "description": "Character[1] walks into the room. Dialogue(Character[1]): 'Hello!'"
    }
  ]
}
```
The system automatically matches characters based on `Character[ID]` and counts dialogue words to determine the scene duration.

---

## Configuration

Edit the following API keys directly in `pipeline.py`:

```python
DASHSCOPE_API_KEY = "YOUR_DASHSCOPE_KEY"
GEMINI_API_KEY = "YOUR_GEMINI_KEY"
IMGBB_API_KEY = "YOUR_IMGBB_KEY"
```

Optional pipeline settings:

```python
MODEL_NAME = "wan2.7-i2v"
INPUT_JSON_PATH = "best.json"
BASE_OUTPUT_DIR = "pipeline_generations"
PANEL_COUNT = 6
AUTO_RESUME = False  # Set to True if you want to resume from the last interrupted run
```

---

## How It Works

### 1. Filter Script Data & Calculate Duration
For each panel (1 to `PANEL_COUNT`):
* Finds the corresponding clip in `best.json`.
* Extracts referenced characters (e.g., `Character[1]`).
* **Calculates optimal duration (2-5 seconds):** 
  * *Visual Bonus*: Compares `manga_panel{i}.png` area to a standard 250x250 baseline.
  * *Text Bonus*: Analyzes dialogue word count (assuming ~10 words per second).

### 2. Generate Candidate Videos
* Uploads `panel{i}.png` to ImgBB.
* Sends two independent generation requests to Wan API using the calculated dynamic duration.
* Polls until both `candidate_1.mp4` and `candidate_2.mp4` are successfully generated.

### 3. Gemini Evaluation
The pipeline renames the candidates to `video_A.mp4` and `video_B.mp4` and uploads them to Gemini 3.1 Pro. Gemini evaluates them based on:
* **Visual Realism** (Fewer AI artifacts, logical physics)
* **Physical Commonsense** (Logical actions and environment)
* **Video-Prompt Alignment** (Matches user prompt constraints and directions)
* **Coherency** (Temporal consistency with the previous panel's winning video)

Gemini returns a strict JSON decision (`A_BETTER`, `B_BETTER`, or `COMPARABLE`).

### 4. Final Concatenation
After all panels are processed and evaluated, the pipeline generates a `concat_list.txt` and uses FFmpeg to merge all best clips into `final_story.mp4`.

---

## Running the Pipeline

Ensure all required images (`panelX.png`, `manga_panelX.png`) and `best.json` are in the root directory, then run:

```bash
python pipeline.py
```

---

## Output Structure

Example of a successful run:

```text
pipeline_generations/
└── run_20260504_120000/
    ├── panel_1/
    │   ├── input.png
    │   ├── candidate_1.mp4
    │   ├── candidate_2.mp4
    │   ├── best_panel_1.mp4
    │   ├── filtered_script.json
    │   ├── wan_prompt.txt
    │   ├── gemini_eval_prompt.txt
    │   └── gemini_eval_response.txt
    │
    ├── panel_2/
    └── ...
    │
    ├── concat_list.txt
    └── final_story.mp4
```

---

## Notes

* **FFmpeg** is strictly required for the final concatenation step.
* Gemini evaluation takes time as files must be uploaded and processed in the cloud.
* Wan generation is asynchronous; the script polls Dashscope every 15 seconds.
* If a generation or evaluation fails midway, you can set `AUTO_RESUME = True` and restart the script to continue from where it left off.