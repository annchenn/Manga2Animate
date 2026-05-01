## Setup

### 1. Conda environment

```bash
conda create -n project python=3.12 -y
conda activate project
pip install -r requirements.txt
```

### 2. Google Cloud auth (Vertex AI)

The pipeline uses Vertex AI (`vertexai=True`). Authenticate once:

```bash
gcloud auth application-default login
```

### 3. `.env`

Copy the example file and edit your values:

```bash
cp .env.example .env
```

See [Hyperparameters](#hyperparameters-env) for what each variable means.

### 4. Prepare a target

For each target, create a directory under `target/<name>/` containing:

```
target/<name>/
    manga_page.{png|jpg}              # full manga page 
    ref.{png|jpg}                     # character reference sheet
    manga_panels/
        panel1.png 
        ... 
        panel6.png     # NUM_CLIPS individual panel crops
```

---

## Usage

```bash
conda activate project
python src/run_all.py -t <target>
# or override iteration count:
NUM_ITERATIONS=3 python src/run_all.py -t <target>
python src/run_all.py -t <target> -n 3
```

All input files come from `target/<target>/` and all results land in `results/<target>/`.

target structure:

```
target/<target>/
    manga_panels/
        panel1.png
        panel2.png
        panel3.png
        panel4.png
        panel5.png
        panel6.png
    manga_page.png
    ref.png
```

---

## Output Layout

```
results/<target>/
    character/
        transcript.txt
        character_description.json

    iter<N>/
        scripts/
            script_{1..NUM_SCRIPTS}.json
        bestScript.json                 # carry-forward: copied from iter<N-1>
        storyboard/
            storyboard_{0..2}.{ext}
        bestStoryboard.{ext}            # carry-forward: copied from iter<N-1>
        intermediate/
            character.json              # script-level character review
            clip_{1..NUM_CLIPS}.json    # per-clip review
            feedback.json               # script-level feedback for next iter
                                        #   (carry-forward: copied from iter<N-1>)
            sb_compare_same_<A>_<B>.json
            sb_compare_diff_new<A>_vs_prev.json   # iter 1+ only
            bestStoryboard_meta.json    # {"branch", "winner_idx", "mode_for_next_iter"}
            storyboard_feedback.json    # {"feedbacks": [...]}  length 1-3
```

`branch` ∈ `{iter0_clean, iter0_cyclic, iterN_carry, iterN_1winner, iterN_2winner,
              iterN_3winner_clean, iterN_3winner_cyclic}` — purely for debugging.

`mode_for_next_iter` ∈ `{one, two, three}` — read by `run_all.py` to choose the
prompt for the next `create_script.py` invocation.

---

## Prompt → Step Mapping

| Step | Prompt |
|---|---|
| Character description | `prompts/create_character_description.txt` |
| Iter 0 script | `prompts/create_script_without_feedback.txt` |
| Iter 1+ script (1 sb feedback) | `prompts/create_script_with_one_feedback.txt` |
| Iter 1+ script (2 sb feedbacks) | `prompts/create_script_with_two_feedback.txt` |
| Iter 1+ script (3 sb feedbacks) | `prompts/create_script_with_three_feedback.txt` |
| Script eval — characters | `prompts/character_reviewer.txt` |
| Script eval — clips | `prompts/clip_reviewer.txt` |
| Storyboard generation | `prompts/create_storyboard.txt` |
| Storyboard compare (same script) | `prompts/storyboard_reviewer_same_script.txt` |
| Storyboard compare (different script) | `prompts/storyboard_reviewer_different_script.txt` |
| Cyclic-tie priority order | `prompts/storyboard_tie_order.txt` |

---

## Hyperparameters (.env)

| Variable | Description |
|----------|-------------|
| `TEXT_MODEL` | Gemini model for text generation/evaluation |
| `IMAGE_MODEL` | Gemini model for storyboard image generation |
| `NUM_SCRIPTS` | Number of scripts generated per iteration |
| `NUM_CLIPS` | Number of manga panels per page |
| `NUM_STORYBOARDS` | Number of storyboards generated per best script |
| `NUM_ITERATIONS` | Number of total iterations (Stage 0 + iter 0..N-1); default 3 |
| `EXP1`, `EXP2` | Few-shot example indices |
| `GOOGLE_CLOUD_PROJECT` | Vertex AI project ID |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI region |

---
# Pipeline

```mermaid
flowchart TD
    subgraph PRE["Stage 0: Pre-processing (once)"]
        direction LR
        MG["magi.py"] -->|transcript.txt| CHAR
        CD["create_character_description.py"] -->|character_description.json| CHAR
        CHAR(["results/&lt;target&gt;/character/"])
    end

    PRE --> IT0["iter 0"]

    subgraph IT0["iter 0"]
        direction TB
        CS0["create_script.py mode=without"] -->|N scripts| ES0["evaluate_scripts.py"]
        ES0 -->|bestScript + script feedback| GS0["generate_storyboard.py"]
        GS0 -->|3 storyboards| EV0["evaluate_storyboards.py<br/>same_script C(3,2) pairwise<br/>+ tie_order if cyclic"]
        EV0 -->|bestStoryboard + 1-2 sb feedbacks<br/>mode_for_next_iter| ITN
    end

    subgraph ITN["iter 1..NUM_ITERATIONS-1"]
        direction TB
        CSN["create_script.py mode=one|two|three"] -->|N scripts| ESN["evaluate_scripts.py"]
        ESN -->|bestScript + script feedback| GSN["generate_storyboard.py"]
        GSN -->|3 storyboards| EVN["evaluate_storyboards.py<br/>3x different_script vs prev best"]
        EVN -->|winners_count| BR{"branch"}
        BR -->|0 winners| CARRY["carry-forward<br/>copy prev bestScript / feedback / bestStoryboard<br/>3 diff feedbacks → mode=three"]
        BR -->|1 winner| ONE["winner = that sb<br/>1 diff feedback → mode=one"]
        BR -->|2 winners| TWO["1 same_script tiebreak<br/>1 same + 1 diff feedback → mode=two"]
        BR -->|3 winners| THREE["like iter 0:<br/>same_script C(3,2) + tie_order<br/>1-2 same feedbacks → mode=one|two"]
    end
```