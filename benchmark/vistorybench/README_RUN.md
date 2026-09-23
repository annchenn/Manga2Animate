# ViStoryBench 執行說明

```bash
cd /project/annchen/project/benchmark/vistorybench
```

所有指令都要在這個目錄下執行（`config.yaml` 用的是相對路徑）。

---

## 1. 建立 conda 環境

```bash
conda create -n vistorybench python=3.11
conda activate vistorybench

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 \
    -c pytorch -c nvidia

# 必要：上一行會把 mkl 裝成 2025、numpy 裝成 2.x，兩個都會讓 torch 2.4 掛掉
conda install "mkl=2023.1.0" "numpy=1.26.4"

pip install -r requirements.txt
```

驗證：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 應該輸出：2.4.0 True
```

如果出現 `undefined symbol: iJIT_NotifyEvent`，就是 mkl 那行沒跑。

## 2. 資料放哪裡

### 已經準備好的（不用動）
我們的劇本轉換成vistorybench放在 `data/dataset/ViStory/<num>/`

ex: `data/dataset/ViStory/03/` 是 goblin 這則故事，已經轉成 ViStoryBench 格式：

```
data/dataset/ViStory/03/
├── story.json                    6 個 shot、2 個角色、中英雙語
└── image/
    ├── boy1/00.jpg               角色參考圖
    └── monster1/00.jpg
```

`data/pretrain/` 的模型權重（9 GB）也已經下載完成。

story id 對照見 `data/dataset/ViStory/story_index.json`：
`01` elf2、`02` elf3、`03` goblin、`04` mado、`05` LoveHina、`06` DualJustice1、
`07` DualJustice2、`08` kick。

`05`–`08` 的 `story.json` 是用 `src/convert_to_vistory.py` 轉出來的，但
`image/<角色>/` 還是空的 —— **角色參考圖要人工放**（`01`–`04` 那幾張是從漫畫頁裁的）。
沒有參考圖的 story，CIDS 會直接 assert 失敗，所以那四個現在還跑不了。

### OCCM 要用的漫畫 panel

OCCM 現在是拿「原始漫畫 panel 裡偵測到幾個角色」跟「生成畫面偵測到幾個」比，
所以每個 story 底下要有 `manga_panels/panel1.png ... panel6.png`：

```bash
python src/link_manga_panels.py          # 從 ../../target_test/<name>/manga_panels/ 複製過來
```

沒放的 story 會退回用劇本的角色數，並且在 log 印警告。

### 要放的：6 張 storyboard

```
data/outputs/<method>/<mode>/en/<timestamp>/03/shots/shot_1.png
                                                     shot_2.png
                                                     ...
                                                     shot_6.png
```

例如：

```
data/outputs/mybaseline/base/en/20260807_120000/03/shots/shot_1.png ... shot_6.png
```

- `<method>`：baseline 名稱。**storyboard 和 video 各自一個 method 資料夾**，`<mode>` 一律 `base`：

  | | storyboard | video |
  |---|---|---|
  | 我們 | `ours_scene` | `ours_video` |
  | AniMaker | `anim_scene` | `anim_video` |

  （新的 baseline 照 `<名字>_scene` / `<名字>_video` 命名就好）
- `<mode>`：一律 `base`
- `<timestamp>`：格式必須是 `YYYYMMDD_HHMMSS`
- `03`：story id，見上面的對照表

影片的 `shots/` 長得不一樣（每個 shot 一個資料夾、裡面 10 張 frame），見 3.1。

可以不用自己想 timestamp，直接用 `date` 產生（把 `mybaseline` 和 `base` 換成你的 method 和 mode）：

```bash
DST=data/outputs/mybaseline/base/en/$(date +%Y%m%d_%H%M%S)/03/shots
mkdir -p "$DST"
```

**檔名規則：**

- 檔名結尾必須是數字。`shot_1.png`、`1.png` 可以；`panel1.png` **不行**，會直接讓程式 crash
- 編號 1–6，要對應 `story.json` 裡的 shot 順序

（如果之後要換角色參考圖，**必須是 `.jpg`**（CIDS 只讀 jpg）。`run_bench.sh` 會自動轉，或手動跑 `python src/normalize_ref_images.py`。）

## 3. 怎麼跑

```bash
conda activate vistorybench
./run_bench.sh --method mybaseline --mode base
```

`run_bench.sh` 會自動轉檔、跑評測、產生 `result.md`。

預設只跑 **cids、csd、prompt_align**（OCCM 包在 cids 裡）。aesthetic / diversity 要自己指定：

```bash
./run_bench.sh --method mybaseline --mode base --metrics cids csd aesthetic diversity
```

**指定 GPU：** 預設用 0 號卡。要換卡就在前面加 `GPU=<編號>`，先用 `nvidia-smi` 看哪張是空的。

```bash
GPU=1 ./run_bench.sh --method mybaseline --mode base
```

- 每次跑會掃過 dataset 裡所有 story，`summary.json` 是「有評到的 story」取平均。
  目前 `01`–`04` 有 outputs，`05`–`08` 缺角色參考圖所以會被跳過。

**參數說明：**

| 參數 | 說明 |
|---|---|
| `--method` | 必填，你的 outputs 資料夾名稱，可以填多個 |
| `--mode` | 建議填，不填會把所有 mode 都列舉一遍 |
| `--metrics` | 可選，預設 `cids csd prompt_align`。另外還有 `aesthetic` `diversity` |
| `--split lite` | **不要用**，lite 名單裡沒有 `03`，會整個跳過 |

### judge 一律用 gpt-4.1

`prompt_align`（以及 CIDS 裡的 PA IA 子項）走 VLM。**judge 固定是 `gpt-4.1`**，設定在
`benchmark/vistorybench/.env`：

```
BASE_URL=https://api.openai.com/v1
MODEL_ID=gpt-4.1
API_KEY=sk-...                 # 優先權高於專案根目錄 .env 的 GEMINI_API_KEY
VLM_TOKENS_PER_MINUTE=30000    # 節流，見下
VLM_TOKENS_PER_CALL=1500
```

**`evaluation_tsv/` 裡現有的每一個 PA 數字都是 gpt-4.1 評的**，換 judge 會讓整欄的分數
改變，跟既有的列不能放在同一張表。所以：要換 judge 就整張表一起重評，不要只換一個 run。

key 的解析順序是 `--api_key` → `API_KEY` → `GEMINI_API_KEY`
（[base_evaluator.py:44](vistorybench/bench/base_evaluator.py:44)）。因為根目錄的 `.env` 有
`GEMINI_API_KEY`，**只要這裡的 `API_KEY` 沒設，就會自動掉回 Gemini 而且不會有任何警告** —
這正是 2026-09-05 那次誤用 Gemini 跑掉一整批的原因。

#### 節流是必要的，不是可選的

這個 org 的 gpt-4.1 只有 **30000 TPM**，每次評分約 1500 token，等於**一分鐘只能 20 次**。
`config.yaml` 的 32 個 worker 一起打只會製造 429 風暴，反而比節流還慢。
[gptv_utils.py 的 `_RateLimiter`](vistorybench/bench/prompt_align/gptv_utils.py) 讀
`VLM_TOKENS_PER_MINUTE` 把每個請求隔開；設成 `0` 才關閉。啟動時會印一行確認：

```
VLM rate limit: 30000 TPM / 1500 tokens per call -> 20.0 calls/min (3.00s apart)
```

**沒看到這行就是沒節流。**

`VLM_TOKENS_PER_CALL` 只是估計值，估低了還是會撞 429，所以限速是自適應的：每次 429 就把
間隔放寬 1.5 倍（上限 30 秒），連續 25 次正常回應再慢慢收回設定值，最後會收斂到端點實際
允許的速率。429 時只會等 `Retry-After` 指定的時間並重送同一個請求，不會像其他錯誤那樣調高
temperature —— 重複評分取樣的條件必須固定。

> 設定是**第一次呼叫時**才讀，不是 import 時。`bench_run.py` 先 import evaluator 才
> `load_dotenv()`，所以在 import 時讀環境變數會什麼都讀不到、限速靜靜失效 —— 之前寫在
> `.env` 的 `VLM_TOKENS_PER_MINUTE` 從來沒生效過，只有在 shell 裡 `export` 才有用。

估時間就用這個上限：每個 story 的呼叫數 ÷ 20 = 分鐘數。

| | 每個 story 的 VLM 呼叫 | 一個 run（8 story） |
|---|---|---|
| storyboard | 90 (PA) + ~40 (IA) | ~1 小時 |
| video（每 shot 10 frame） | 900 (PA) + ~40 (IA) | ~8 小時 |

其他 metric（CSD / CIDS cross-self / OCCM）純本地模型，不需要 API 也不花錢。要只跑這些，
把金鑰拿掉即可：

```bash
env -u API_KEY -u GEMINI_API_KEY python bench_run.py --language en \
    --method <method> --mode base --metrics csd cids
```

## 3.1 影片：一個 shot 抽 10 張 frame

影片的每個 shot 會抽 10 張 frame，每張各自評分再平均。目錄長這樣：

```
<timestamp>/<story_id>/shots/shot_1/frame_00.png ... frame_09.png
                             shot_2/...
```

- 我們自己的 pipeline（一支影片塞 6 個 shot）：
  `python ../../src/crop_video_frames.py -t goblin --frames-per-shot 10`
  先偵測切點，再在每一段內均勻取樣。
- 一個 shot 一支影片的 baseline：
  `python src/export_animaker_shots.py --videos-dir <clips> --story 03 --frames-per-shot 10`

`--frames-per-shot 1` 會退回舊的 `shot_1.png ... shot_6.png` 平鋪格式，storyboard 就是這種。
兩種格式 `run_bench.sh` 都吃，單張圖等同「只有一張 frame」。

CIDS 的 self 分數只用每個 shot 的代表 frame（中間那張），cross 才用全部 10 張——
不然同一段影片內的 frame 互比會把 self 灌得很高，跟 storyboard 的數字就不能比了。

## 3.2 這次四個 run 是怎麼跑的（可重現）

```bash
cd /project/annchen/project/benchmark/vistorybench
PY=/home_nfs/annchen/miniconda3/envs/vistorybench/bin/python
```

**(1) 我們的 storyboard** — 圖已經在 outputs 裡，直接評：

```bash
CUDA_VISIBLE_DEVICES=2 $PY bench_run.py --language en \
    --method ours_scene --mode base --timestamp 20260807_000000
```

**(2) AniMaker storyboard** — 原本四個 story 散在四個 timestamp，先併成一個再評：

```bash
CUDA_VISIBLE_DEVICES=2 $PY bench_run.py --language en \
    --method anim_scene --mode base --timestamp 20260828_all
```

**(3) 我們的 video** — 先切影片（一支 15s 影片內含 6 個 shot，要偵測切點）再評：

```bash
$PY ../../src/crop_video_frames.py -t elf2 -t elf3 -t goblin -t mado \
    --timestamp 20260828_10frames --frames-per-shot 10
CUDA_VISIBLE_DEVICES=2 $PY bench_run.py --language en \
    --method ours_video --mode base --timestamp 20260828_10frames
```

**(4) AniMaker video** — 一個 shot 一支 clip，不用偵測切點：

```bash
SRC=../../evaluation_results/animaker_new/vistory_outputs/source_video
for pair in "elf2 01" "elf3 02" "goblin 03" "mado 04"; do
    set -- $pair
    $PY src/export_animaker_shots.py --videos-dir $SRC/$1 --story $2 \
        --method anim_video --mode base --timestamp 20260828_10frames --frames-per-shot 10
done
CUDA_VISIBLE_DEVICES=2 $PY bench_run.py --language en \
    --method anim_video --mode base --timestamp 20260828_10frames
```

一個 story 大約 11 分鐘（CIDS 的 VLM 呼叫佔最久），四個 run 全跑完約 5.5 小時。
跑完用 `python src/make_result_md.py --run <run 目錄>` 產 `result.md`
（直接呼叫 `bench_run.py` 不會自動產，走 `run_bench.sh` 才會）。

## 4. 產物

```
data/bench_results/<method>/<mode>/en/<timestamp>/
├── result.md              ← 看這個，所有數值整理成表格
├── summary.json           所有 metric 的最終數值
├── metadata.json          這次執行的 method / mode / language / timestamp
├── manifest.json          每個 metric × story 的完成狀態，用於續跑
└── metrics/
    ├── cids/story_results.json        逐角色、逐 shot 分數
    ├── csd/story_results.json         逐 shot 風格分數
    ├── aesthetic/story_results.json   逐圖美學分數
    └── diversity/scores.json          Inception Score
```

## 5. AniMaker baseline 的 shot 圖是怎麼抽出來的

我們自己的 storyboard 影片一支裡塞了 6 個 shot 首尾相接，所以 `../../src/crop_video_frames.py`
要先偵測鏡頭切點才能切出 6 張代表幀。AniMaker（我們拿來對照的 baseline，上游
[HITsz-TMG/Anim-Director](https://github.com/HITsz-TMG/Anim-Director)，論文
[arXiv:2506.10540](https://arxiv.org/abs/2506.10540)）的產物結構不一樣：它的 pipeline
本來就是**一個 shot 一支獨立影片**，所以完全不用切點偵測，每支影片直接抽一幀就是一個 shot。

腳本：`src/export_animaker_shots.py`（沿用 AniMaker 自己的
`export_vistory_shots.py::_extract_frame` 邏輯搬過來的，抽出的畫面逐 pixel 核對過完全一致）。

**需要什麼資料**：只需要 AniMaker 該 story 選定（chosen）的 shot 影片檔案本身，檔名維持它原本
的命名慣例 `scene_<scene_pos>_clip_<clip_id>.mp4`（AniMaker 產物路徑通常在它自己
repo 的 `Pipeline/res/<story>/videos/chosen/` 底下），全部丟進同一個資料夾即可——**不需要**
附帶 `script.json`/`pipeline.json` 或其他 pipeline 中間產物。shot 編號是從檔名裡的
`(scene_pos, clip_id)` 排序後累加得出，不是讀 story.json。

**抽幾幀**：預設每支 clip 均勻抽 **10 張**（`--frames-per-shot 10`），跟我們自己的影片一致。
`--frames-per-shot 1` 才會退回舊行為：一支 clip 抽一張中間幀（`--frame mid`，也支援
`first`/`last`），中間幀最能代表整支片段實際演出的畫面，能避開片頭/片尾轉場的過渡幀。

**怎麼執行**：

```bash
python src/export_animaker_shots.py \
    --videos-dir /path/to/animaker_chosen_clips \
    --story 03
```

輸出：

```
data/outputs/anim_video/base/en/<timestamp>/03/shots/shot_1/frame_00.png ... frame_09.png
                                                      shot_2/...
```

（`--frames-per-shot 1` 則是 `shots/shot_1.png ... shot_6.png`）

`--method`（預設 `anim_video`）/`--mode`（預設 `base`）/`--language`（預設 `en`）/
`--outputs-root`（預設本資料夾內的 `data/outputs`）都可以覆寫，用法跟第 2 節的
`<method>/<mode>/<lang>/<timestamp>` 慣例一致。接下來跟第 3 節一樣：把這棵樹複製到實際跑
`run_bench.sh` 的位置，再執行：

```bash
./run_bench.sh --method anim_video --mode base
```

## 6. `result.md` 欄位對照

**metric 意義**（`result.md` 的欄位順序與論文表格一致）：

| 論文欄位 | 對應的 key | 意義 |
|---|---|---|
| CSD Cross / Self | `cross_csd` / `self_csd` | 風格一致性 |
| CIDS Cross / Self | `cids_cross_mean` / `cids_self_mean` | 角色一致性。cross 是跟參考圖比對，self 是生成圖之間比對 |
| PA Scene | `scene` | 背景場景是否符合腳本 |
| PA Shot | `camera` | 鏡頭語言是否符合腳本 |
| PA CI | `character_action` | 畫面主要角色動作是否符合腳本 |
| PA IA | `single_character_action` | 指定角色的個別動作是否符合（CIDS 內部子項，需要 VLM） |
| PA Avg. | — | 上面四項平均 |
| CM | `occm` | 生成畫面的角色數量 vs **原始漫畫 panel** 偵測到的角色數量 |
| Inc | `inception_score` | Inception Score，生成圖多樣性。圖太少時數值不穩定 |
| Aes | `aesthetic_score` | 美學評分 |
| CP | `copy_paste_score` | 越低越好，高代表直接貼參考圖而不是重畫 |

## 7. 這一版改了什麼

對照 `note` 那六條：

1. **PA 每項評 5 次**，回報平均和標準差（`scene` / `scene_std` …）。重複評分用
   `top_p=1.0`、`temperature=1.0`；原本的 `top_p=0.2` 會讓 5 次結果完全一樣、std 恆為 0。
   次數和溫度在 `config.yaml` 的 `evaluators.prompt_align` 底下調。
2. **OCCM 改用 GroundingDINO 偵測漫畫 panel** 當作參考角色數（原本是讀劇本的角色列表），
   panel 和生成畫面都用 prompt `"character"`、threshold 0.25。
3. **CIDS 的角色相似度改用 SigLIP**（`data/pretrain/google/siglip-so400m-patch14-384`）。
   `config.yaml` 的 `evaluators.cids.encoders.override` 可以改回 `clip`。
4. **影片每個 shot 抽 10 張 frame 評分再平均**，見 3.1。
5. **預設只跑 CSD / CIDS / PA / OCCM**。
6. dataset 補到八個 story（`05`–`08` 還缺角色參考圖）。

跑出來的 `summary.json` 多了 `*_std` 欄位，`result.md` 的 PA 會印成 `4.20 ±0.45`。
每個 story 的 `story_results.json` 還有 `occm_details`（每個 shot 期望 vs 偵測到幾個角色）
和 `scoring`（這次用的 encoder、threshold、重複次數）。

`utils/build_evaluation_sheet.py` 會把所有 run 收成 paper 格式的表：

```bash
# 兩個分頁（Storyboard / Video），ViStoryBench 的指標欄有底色
/home_nfs/annchen/miniconda3/envs/vistorybench/bin/python utils/build_evaluation_sheet.py \
    -o /tmp/evaluation_table.xlsx --markdown
```

`--markdown` 會順便把表印在 terminal。`.csv` 副檔名則產單一平面表（沒有分頁和顏色）。
沒跑過的 baseline 會是 `—`，欄位先擺著。要新增 baseline 或指標就改檔案最上面的
`BLOCKS` / `METRIC_SOURCES` / `COLUMNS`。
