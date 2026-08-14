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

story id 對照：`01` = elf2、`02` = elf3、`03` = **goblin**、`04` = mado

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

- `<method>`：你的 baseline 名稱，自己取，不要蓋掉 `ours/`
- `<mode>`：隨意，`base` 即可
- `<timestamp>`：格式必須是 `YYYYMMDD_HHMMSS`
- `03`：story id，goblin

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

只跑特定 metric：

```bash
./run_bench.sh --method mybaseline --mode base --metrics cids csd
```

**指定 GPU：** 預設用 0 號卡。要換卡就在前面加 `GPU=<編號>`，先用 `nvidia-smi` 看哪張是空的。

```bash
GPU=1 ./run_bench.sh --method mybaseline --mode base
```

- 每次跑他會掃過那 <method><base>的dataset的所有story，   `summary.json` 是所有有評到的 story 取平均，現在只有03(goblin)有放所以他只會用03的

**參數說明：**

| 參數 | 說明 |
|---|---|
| `--method` | 必填，你的 outputs 資料夾名稱，可以填多個 |
| `--mode` | 建議填，不填會把所有 mode 都列舉一遍 |
| `--metrics` | 可選，預設全跑。可用：`cids` `csd` `aesthetic` `diversity` `prompt_align` |
| `--split lite` | **不要用**，lite 名單裡沒有 `03`，會整個跳過 |

`prompt_align` 走 VLM，設定在 `.env`（`API_KEY` / `BASE_URL` / `MODEL_ID`，目前指向 Gemini 的 OpenAI 相容端點）。其他四個 metric 純本地模型，不需要 API。

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
| CM | `occm` | 畫面中角色數量是否符合腳本 |
| Inc | `inception_score` | Inception Score，生成圖多樣性。圖太少時數值不穩定 |
| Aes | `aesthetic_score` | 美學評分 |
| CP | `copy_paste_score` | 越低越好，高代表直接貼參考圖而不是重畫 |

