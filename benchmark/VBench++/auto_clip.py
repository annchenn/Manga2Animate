import os
import cv2
import numpy as np
from itertools import combinations

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ============================================================
# 設定
# ============================================================

NEED_CLIP_DIR = "need_clip"

# 固定 6 panels → 5 transitions
NUM_PANELS = 6
NUM_CUTS = NUM_PANELS - 1

# ------------------------------------------------------------
# True：
#   不管資料夾是否已經有完整 6 個 panel，都重新切割
#
# False：
#   如果已經有完整 panel_01 ~ panel_06，就跳過
# ------------------------------------------------------------

FORCE_RECUT = False


# ============================================================
# Global candidate peak 設定
# ============================================================

# candidate peaks 彼此最少間隔
CANDIDATE_DISTANCE_SECONDS = 0.20

# 最終任何兩個 transition 至少相隔多久
MIN_PANEL_SECONDS = 0.50

# transition 不允許太靠近影片頭尾
EDGE_SECONDS = 0.40

# Global frame difference smoothing
SMOOTH_SIGMA = 1.5

# 最多留下多少 candidate peaks 做組合搜尋
MAX_CANDIDATES = 30


# ============================================================
# Persistent transition 分析
# ============================================================

# Global candidate 前後分析多少秒
TRANSITION_WINDOW_SECONDS = 0.35

# 為了加速分析，縮小影像寬度
ANALYSIS_WIDTH = 320


# ============================================================
# Global transition score 權重
# ============================================================

# 單一 frame change
WEIGHT_LOCAL_DIFF = 0.15

# peak prominence
WEIGHT_PROMINENCE = 0.20

# transition 前後平均畫面的差異
WEIGHT_CROSS_CHANGE = 0.40

# transition 前後差異 / 內部 motion
WEIGHT_STABILITY_RATIO = 0.25


# ============================================================
# Local boundary refinement 設定
# ============================================================

# 從 global transition 往前搜尋多少秒
REFINE_LOOKBACK_SECONDS = 0.40

# global transition 後方仍稍微搜尋
REFINE_LOOKAHEAD_SECONDS = 0.12

# 用更前面的區域估計正常 motion baseline
REFINE_BASELINE_SECONDS = 0.35

# transition peak 強度的多少比例視為 onset threshold
#
# 越小：
#   越容易把 boundary 往前拉
#
# 越大：
#   越接近 transition peak
REFINE_PEAK_FRACTION = 0.25

# baseline + MAD * multiplier
REFINE_MAD_MULTIPLIER = 2.5

# 往前搜尋時，連續多少個 frame 低於 threshold
# 才視為 transition 已經結束
REFINE_QUIET_FRAMES = 2

# refinement 使用比較輕微 smoothing
REFINE_SMOOTH_SIGMA = 0.5


# ============================================================
# 支援格式
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


# ============================================================
# 判斷是否已經完整切割
# ============================================================

def already_clipped(output_dir):

    if not os.path.isdir(output_dir):
        return False

    for i in range(1, NUM_PANELS + 1):

        path = os.path.join(
            output_dir,
            f"panel_{i:02d}.mp4"
        )

        if not os.path.isfile(path):
            return False

        if os.path.getsize(path) == 0:
            return False

    return True


# ============================================================
# 清除舊輸出
# ============================================================

def clean_output(output_dir):

    if not os.path.isdir(output_dir):
        return

    for filename in os.listdir(output_dir):

        if (
            filename.startswith("panel_")
            and filename.lower().endswith(".mp4")
        ):

            path = os.path.join(
                output_dir,
                filename
            )

            try:
                os.remove(path)

            except Exception as e:

                print(
                    f"警告：無法刪除 {path}"
                )

                print(
                    f"      {e}"
                )

    txt_path = os.path.join(
        output_dir,
        "transitions.txt"
    )

    if os.path.isfile(txt_path):

        try:
            os.remove(txt_path)

        except Exception:
            pass


# ============================================================
# Resize frame
# ============================================================

def resize_for_analysis(frame):

    height, width = frame.shape[:2]

    if width <= ANALYSIS_WIDTH:

        resized = frame

    else:

        scale = (
            ANALYSIS_WIDTH / width
        )

        new_height = int(
            round(
                height * scale
            )
        )

        resized = cv2.resize(
            frame,
            (
                ANALYSIS_WIDTH,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    return gray


# ============================================================
# 讀影片 + frame difference
# ============================================================

def load_video_analysis(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"無法開啟影片：{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    metadata_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        cap.release()

        raise RuntimeError(
            f"無法取得 FPS：{video_path}"
        )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = resize_for_analysis(
            frame
        )

        frames.append(
            gray
        )

    cap.release()

    if len(frames) < 2:

        raise RuntimeError(
            "影片 frame 數量不足"
        )

    total_frames = len(frames)

    if metadata_frames != total_frames:

        print(
            f"注意：metadata frames = "
            f"{metadata_frames}, "
            f"實際讀取 = {total_frames}"
        )

    # --------------------------------------------------------
    # 相鄰 frame difference
    # --------------------------------------------------------

    frame_diff = np.zeros(
        total_frames,
        dtype=np.float64
    )

    for i in range(
        1,
        total_frames
    ):

        diff = cv2.absdiff(
            frames[i],
            frames[i - 1]
        )

        frame_diff[i] = float(
            np.mean(diff)
        )

    return (
        frames,
        frame_diff,
        fps,
        total_frames
    )


# ============================================================
# Robust normalization
# ============================================================

def robust_normalize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:
        return values

    low = np.percentile(
        values,
        5
    )

    high = np.percentile(
        values,
        95
    )

    if high <= low + 1e-8:

        return np.zeros_like(
            values
        )

    normalized = (
        values - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0.0,
        1.0
    )


# ============================================================
# 找 candidate peaks
# ============================================================

def find_candidate_peaks(
    frame_diff,
    fps
):

    smooth_diff = gaussian_filter1d(
        frame_diff,
        sigma=SMOOTH_SIGMA
    )

    distance = max(
        1,
        int(
            round(
                CANDIDATE_DISTANCE_SECONDS
                * fps
            )
        )
    )

    edge_frames = max(
        1,
        int(
            round(
                EDGE_SECONDS
                * fps
            )
        )
    )

    peaks, properties = find_peaks(
        smooth_diff,
        distance=distance,
        prominence=0
    )

    prominences = properties[
        "prominences"
    ]

    valid = (
        (peaks >= edge_frames)
        &
        (
            peaks
            <
            len(frame_diff)
            -
            edge_frames
        )
    )

    peaks = peaks[
        valid
    ]

    prominences = prominences[
        valid
    ]

    return (
        peaks,
        prominences,
        smooth_diff
    )


# ============================================================
# 計算 candidate transition 特徵
# ============================================================

def calculate_candidate_features(
    frames,
    smooth_diff,
    peaks,
    prominences,
    fps
):

    n = len(frames)

    window = max(
        2,
        int(
            round(
                TRANSITION_WINDOW_SECONDS
                * fps
            )
        )
    )

    feature_list = []

    for peak, prominence in zip(
        peaks,
        prominences
    ):

        peak = int(
            peak
        )

        before_start = (
            peak - window
        )

        before_end = peak

        after_start = peak

        after_end = (
            peak + window
        )

        if before_start < 0:
            continue

        if after_end > n:
            continue

        before_frames = frames[
            before_start:
            before_end
        ]

        after_frames = frames[
            after_start:
            after_end
        ]

        if (
            len(before_frames) < 2
            or
            len(after_frames) < 2
        ):
            continue

        # ----------------------------------------------------
        # Transition 前平均畫面
        # ----------------------------------------------------

        before_mean = np.mean(
            np.stack(
                before_frames,
                axis=0
            ),
            axis=0
        )

        # ----------------------------------------------------
        # Transition 後平均畫面
        # ----------------------------------------------------

        after_mean = np.mean(
            np.stack(
                after_frames,
                axis=0
            ),
            axis=0
        )

        # ----------------------------------------------------
        # Cross change
        # ----------------------------------------------------

        cross_change = float(
            np.mean(
                np.abs(
                    after_mean
                    -
                    before_mean
                )
            )
        )

        # ----------------------------------------------------
        # Transition 前內部 motion
        # ----------------------------------------------------

        before_motion_values = []

        for j in range(
            len(before_frames) - 1
        ):

            d = cv2.absdiff(
                before_frames[j],
                before_frames[j + 1]
            )

            before_motion_values.append(
                float(
                    np.mean(d)
                )
            )

        before_motion = float(
            np.mean(
                before_motion_values
            )
        )

        # ----------------------------------------------------
        # Transition 後內部 motion
        # ----------------------------------------------------

        after_motion_values = []

        for j in range(
            len(after_frames) - 1
        ):

            d = cv2.absdiff(
                after_frames[j],
                after_frames[j + 1]
            )

            after_motion_values.append(
                float(
                    np.mean(d)
                )
            )

        after_motion = float(
            np.mean(
                after_motion_values
            )
        )

        internal_motion = (
            before_motion
            +
            after_motion
        ) / 2.0

        # ----------------------------------------------------
        # Stability ratio
        # ----------------------------------------------------

        stability_ratio = (
            cross_change
            /
            (
                internal_motion
                +
                1.0
            )
        )

        local_diff = float(
            smooth_diff[
                peak
            ]
        )

        feature_list.append({
            "frame": peak,
            "local_diff": local_diff,
            "prominence": float(
                prominence
            ),
            "cross_change": cross_change,
            "internal_motion": internal_motion,
            "stability_ratio": stability_ratio,
        })

    return feature_list


# ============================================================
# Transition score
# ============================================================

def calculate_transition_scores(
    feature_list
):

    if len(feature_list) == 0:
        return feature_list

    local = robust_normalize([
        x["local_diff"]
        for x in feature_list
    ])

    prominence = robust_normalize([
        x["prominence"]
        for x in feature_list
    ])

    cross = robust_normalize([
        x["cross_change"]
        for x in feature_list
    ])

    ratio = robust_normalize([
        x["stability_ratio"]
        for x in feature_list
    ])

    for i, item in enumerate(
        feature_list
    ):

        score = (
            WEIGHT_LOCAL_DIFF
            *
            local[i]

            +

            WEIGHT_PROMINENCE
            *
            prominence[i]

            +

            WEIGHT_CROSS_CHANGE
            *
            cross[i]

            +

            WEIGHT_STABILITY_RATIO
            *
            ratio[i]
        )

        item[
            "transition_score"
        ] = float(
            score
        )

    return feature_list


# ============================================================
# 從 candidates 中選最佳 5 個 global transitions
# ============================================================

def select_best_five(
    candidates,
    total_frames,
    fps
):

    if len(candidates) < NUM_CUTS:

        raise RuntimeError(
            f"candidate 只有 "
            f"{len(candidates)} 個，"
            f"不足 {NUM_CUTS} 個。"
        )

    # --------------------------------------------------------
    # 先取 transition score 較好的候選
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x:
        x["transition_score"],
        reverse=True
    )

    candidates = candidates[
        :MAX_CANDIDATES
    ]

    # 重新依照時間排序
    candidates = sorted(
        candidates,
        key=lambda x:
        x["frame"]
    )

    min_panel_frames = max(
        1,
        int(
            round(
                MIN_PANEL_SECONDS
                * fps
            )
        )
    )

    best_combo = None

    best_score = (
        -np.inf
    )

    # --------------------------------------------------------
    # 嘗試所有 5-cut 組合
    # --------------------------------------------------------

    for combo in combinations(
        candidates,
        NUM_CUTS
    ):

        frames = [
            x["frame"]
            for x in combo
        ]

        boundaries = [
            0,
            *frames,
            total_frames
        ]

        segment_lengths = np.diff(
            boundaries
        )

        # 有任何一段太短就不要
        if np.any(
            segment_lengths
            <
            min_panel_frames
        ):
            continue

        # 五個 transition score 加總
        score = sum(
            x["transition_score"]
            for x in combo
        )

        # ----------------------------------------------------
        # 避免極端長 panel
        #
        # 只在某一段超過整支影片 45% 時稍微處罰
        # 不會強迫六等分
        # ----------------------------------------------------

        longest_ratio = (
            np.max(
                segment_lengths
            )
            /
            total_frames
        )

        if longest_ratio > 0.45:

            score -= (
                longest_ratio
                -
                0.45
            ) * 2.0

        if score > best_score:

            best_score = score

            best_combo = combo

    if best_combo is None:

        raise RuntimeError(
            "無法找到合法的 5 個 global transitions"
        )

    return sorted(
        best_combo,
        key=lambda x:
        x["frame"]
    )


# ============================================================
# Local boundary refinement
# ============================================================

def refine_transition_boundaries(
    selected,
    frame_diff,
    fps,
    total_frames
):

    # --------------------------------------------------------
    # Refinement 使用較輕微 smoothing
    # --------------------------------------------------------

    refine_signal = gaussian_filter1d(
        frame_diff,
        sigma=REFINE_SMOOTH_SIGMA
    )

    lookback = max(
        2,
        int(
            round(
                REFINE_LOOKBACK_SECONDS
                * fps
            )
        )
    )

    lookahead = max(
        1,
        int(
            round(
                REFINE_LOOKAHEAD_SECONDS
                * fps
            )
        )
    )

    baseline_length = max(
        2,
        int(
            round(
                REFINE_BASELINE_SECONDS
                * fps
            )
        )
    )

    refined_items = []

    # --------------------------------------------------------
    # 每個 global transition 個別微調
    # --------------------------------------------------------

    for item in selected:

        original_frame = int(
            item["frame"]
        )

        search_start = max(
            1,
            original_frame
            -
            lookback
        )

        search_end = min(
            total_frames - 1,
            original_frame
            +
            lookahead
        )

        # ----------------------------------------------------
        # Baseline 使用搜尋區間之前
        # ----------------------------------------------------

        baseline_end = (
            search_start
        )

        baseline_start = max(
            1,
            baseline_end
            -
            baseline_length
        )

        baseline_values = refine_signal[
            baseline_start:
            baseline_end
        ]

        if len(
            baseline_values
        ) == 0:

            baseline_values = (
                refine_signal[
                    max(
                        1,
                        search_start - 3
                    ):
                    search_start
                ]
            )

        # baseline 還是沒有資料
        if len(
            baseline_values
        ) == 0:

            new_item = item.copy()

            new_item[
                "original_frame"
            ] = original_frame

            new_item[
                "frame"
            ] = original_frame

            new_item[
                "refine_reverted"
            ] = True

            refined_items.append(
                new_item
            )

            continue

        baseline = float(
            np.median(
                baseline_values
            )
        )

        mad = float(
            np.median(
                np.abs(
                    baseline_values
                    -
                    baseline
                )
            )
        )

        # ----------------------------------------------------
        # Local window 最大 change
        # ----------------------------------------------------

        local_values = refine_signal[
            search_start:
            search_end + 1
        ]

        local_peak_offset = int(
            np.argmax(
                local_values
            )
        )

        local_peak_frame = (
            search_start
            +
            local_peak_offset
        )

        local_peak_value = float(
            refine_signal[
                local_peak_frame
            ]
        )

        # ----------------------------------------------------
        # Adaptive threshold
        # ----------------------------------------------------

        peak_threshold = (
            baseline
            +
            REFINE_PEAK_FRACTION
            *
            max(
                0.0,
                local_peak_value
                -
                baseline
            )
        )

        mad_threshold = (
            baseline
            +
            REFINE_MAD_MULTIPLIER
            *
            mad
        )

        threshold = max(
            peak_threshold,
            mad_threshold
        )

        # ----------------------------------------------------
        # 從 local peak 往前找 transition onset
        # ----------------------------------------------------

        onset = (
            local_peak_frame
        )

        quiet_count = 0

        for frame_idx in range(
            local_peak_frame - 1,
            search_start - 1,
            -1
        ):

            value = refine_signal[
                frame_idx
            ]

            if value >= threshold:

                onset = frame_idx

                quiet_count = 0

            else:

                quiet_count += 1

                if (
                    quiet_count
                    >=
                    REFINE_QUIET_FRAMES
                ):

                    break

        refined_frame = int(
            onset
        )

        new_item = item.copy()

        new_item[
            "original_frame"
        ] = original_frame

        new_item[
            "refined_peak_frame"
        ] = local_peak_frame

        new_item[
            "refine_baseline"
        ] = baseline

        new_item[
            "refine_threshold"
        ] = threshold

        new_item[
            "refine_peak_value"
        ] = local_peak_value

        new_item[
            "frame"
        ] = refined_frame

        new_item[
            "refine_reverted"
        ] = False

        refined_items.append(
            new_item
        )

    # --------------------------------------------------------
    # 按 global transition 順序整理
    # --------------------------------------------------------

    refined_items = sorted(
        refined_items,
        key=lambda x:
        x["original_frame"]
    )

    # --------------------------------------------------------
    # Safety check
    #
    # refinement 後如果造成 panel 太短，
    # 該 transition 退回 global frame
    # --------------------------------------------------------

    min_panel_frames = max(
        1,
        int(
            round(
                MIN_PANEL_SECONDS
                * fps
            )
        )
    )

    # 多次檢查，因為某一個退回之後會影響鄰居
    for _ in range(2):

        current_frames = [
            int(
                x["frame"]
            )
            for x in refined_items
        ]

        boundaries = [
            0,
            *current_frames,
            total_frames
        ]

        segment_lengths = np.diff(
            boundaries
        )

        for i in range(
            len(
                refined_items
            )
        ):

            left_length = (
                segment_lengths[i]
            )

            right_length = (
                segment_lengths[
                    i + 1
                ]
            )

            if (
                left_length
                <
                min_panel_frames

                or

                right_length
                <
                min_panel_frames
            ):

                refined_items[i][
                    "frame"
                ] = int(
                    refined_items[i][
                        "original_frame"
                    ]
                )

                refined_items[i][
                    "refine_reverted"
                ] = True

    # --------------------------------------------------------
    # 最後確保 frame 是遞增
    # --------------------------------------------------------

    final_frames = [
        int(
            x["frame"]
        )
        for x in refined_items
    ]

    if any(
        final_frames[i]
        >=
        final_frames[i + 1]

        for i in range(
            len(final_frames) - 1
        )
    ):

        # 有問題就全部退回 global
        print(
            "警告：refinement 造成切點順序異常，"
            "全部退回 global transitions。"
        )

        for item in refined_items:

            item["frame"] = int(
                item[
                    "original_frame"
                ]
            )

            item[
                "refine_reverted"
            ] = True

    return refined_items


# ============================================================
# 完整 transition detection
# ============================================================

def detect_transitions(
    frames,
    frame_diff,
    fps,
    total_frames
):

    # --------------------------------------------------------
    # Step 1：找大量 candidate peaks
    # --------------------------------------------------------

    (
        peaks,
        prominences,
        smooth_diff
    ) = find_candidate_peaks(
        frame_diff,
        fps
    )

    print(
        f"初始 candidate peaks："
        f"{len(peaks)} 個"
    )

    # --------------------------------------------------------
    # Step 2：分析 candidate 前後畫面
    # --------------------------------------------------------

    features = calculate_candidate_features(
        frames,
        smooth_diff,
        peaks,
        prominences,
        fps
    )

    # --------------------------------------------------------
    # Step 3：算 transition score
    # --------------------------------------------------------

    features = calculate_transition_scores(
        features
    )

    print(
        f"有效 transition candidates："
        f"{len(features)} 個"
    )

    # --------------------------------------------------------
    # Step 4：Global 搜尋最佳 5 個
    # --------------------------------------------------------

    global_best = select_best_five(
        features,
        total_frames,
        fps
    )

    # --------------------------------------------------------
    # Step 5：Local boundary refinement
    # --------------------------------------------------------

    refined = refine_transition_boundaries(
        global_best,
        frame_diff,
        fps,
        total_frames
    )

    return refined


# ============================================================
# OpenCV 切影片
# ============================================================

def cut_video(
    video_path,
    output_dir,
    cut_frames,
    fps,
    total_frames
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"無法開啟影片：{video_path}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    boundaries = [
        0,
        *cut_frames,
        total_frames
    ]

    fourcc = (
        cv2.VideoWriter_fourcc(
            *"mp4v"
        )
    )

    for i in range(
        NUM_PANELS
    ):

        start_frame = int(
            boundaries[i]
        )

        end_frame = int(
            boundaries[i + 1]
        )

        output_path = os.path.join(
            output_dir,
            f"panel_{i + 1:02d}.mp4"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (
                width,
                height
            )
        )

        if not writer.isOpened():

            cap.release()

            raise RuntimeError(
                f"無法建立：{output_path}"
            )

        # ----------------------------------------------------
        # 跳到 panel 起始 frame
        # ----------------------------------------------------

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame
        )

        written = 0

        for _ in range(
            start_frame,
            end_frame
        ):

            ret, frame = cap.read()

            if not ret:
                break

            writer.write(
                frame
            )

            written += 1

        writer.release()

        if written == 0:

            cap.release()

            raise RuntimeError(
                f"Panel {i + 1} "
                f"沒有寫入任何 frame"
            )

        if not os.path.isfile(
            output_path
        ):

            cap.release()

            raise RuntimeError(
                f"沒有產生：{output_path}"
            )

        if os.path.getsize(
            output_path
        ) == 0:

            cap.release()

            raise RuntimeError(
                f"輸出影片為空：{output_path}"
            )

        print(
            f"    Panel {i + 1}: "
            f"frame {start_frame} ~ "
            f"{end_frame - 1} "
            f"| {written} frames "
            f"| "
            f"{start_frame / fps:.6f} ~ "
            f"{end_frame / fps:.6f} sec"
        )

    cap.release()


# ============================================================
# 儲存 transition 資訊
# ============================================================

def save_transition_info(
    output_dir,
    selected,
    fps,
    total_frames
):

    output_path = os.path.join(
        output_dir,
        "transitions.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"FPS: {fps}\n"
        )

        f.write(
            f"Total frames: "
            f"{total_frames}\n\n"
        )

        for i, item in enumerate(
            selected,
            start=1
        ):

            frame = int(
                item["frame"]
            )

            original_frame = int(
                item.get(
                    "original_frame",
                    frame
                )
            )

            f.write(
                f"Transition {i}\n"
            )

            f.write(
                f"Global frame: "
                f"{original_frame}\n"
            )

            f.write(
                f"Refined frame: "
                f"{frame}\n"
            )

            f.write(
                f"Time: "
                f"{frame / fps:.6f} sec\n"
            )

            f.write(
                f"Transition score: "
                f"{item['transition_score']:.6f}\n"
            )

            f.write(
                f"Local diff: "
                f"{item['local_diff']:.6f}\n"
            )

            f.write(
                f"Prominence: "
                f"{item['prominence']:.6f}\n"
            )

            f.write(
                f"Cross change: "
                f"{item['cross_change']:.6f}\n"
            )

            f.write(
                f"Internal motion: "
                f"{item['internal_motion']:.6f}\n"
            )

            f.write(
                f"Stability ratio: "
                f"{item['stability_ratio']:.6f}\n"
            )

            if (
                "refined_peak_frame"
                in item
            ):

                f.write(
                    f"Refinement local peak: "
                    f"{item['refined_peak_frame']}\n"
                )

                f.write(
                    f"Refinement baseline: "
                    f"{item['refine_baseline']:.6f}\n"
                )

                f.write(
                    f"Refinement threshold: "
                    f"{item['refine_threshold']:.6f}\n"
                )

                f.write(
                    f"Refinement peak value: "
                    f"{item['refine_peak_value']:.6f}\n"
                )

            f.write(
                f"Refinement reverted: "
                f"{item.get('refine_reverted', False)}\n"
            )

            f.write("\n")


# ============================================================
# 處理單支影片
# ============================================================

def process_video(
    video_path
):

    filename = os.path.basename(
        video_path
    )

    video_name = os.path.splitext(
        filename
    )[0]

    output_dir = os.path.join(
        NEED_CLIP_DIR,
        video_name
    )

    # --------------------------------------------------------
    # 已經完整切過
    # --------------------------------------------------------

    if (
        not FORCE_RECUT
        and
        already_clipped(
            output_dir
        )
    ):

        print()
        print(
            f"[SKIP] {filename}"
        )

        print(
            "       已完整切成 6 個 panel"
        )

        return "skip"

    # --------------------------------------------------------
    # Output 已存在
    # --------------------------------------------------------

    if os.path.isdir(
        output_dir
    ):

        print()

        if FORCE_RECUT:

            print(
                f"[FORCE RECUT] "
                f"{filename}"
            )

        else:

            print(
                f"[RETRY] "
                f"{filename}"
            )

        clean_output(
            output_dir
        )

    print()
    print("=" * 70)

    print(
        f"[PROCESS] {filename}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # 影片分析
    # --------------------------------------------------------

    (
        frames,
        frame_diff,
        fps,
        total_frames
    ) = load_video_analysis(
        video_path
    )

    duration = (
        total_frames / fps
    )

    print(
        f"FPS          : "
        f"{fps:.6f}"
    )

    print(
        f"Total frames : "
        f"{total_frames}"
    )

    print(
        f"Duration     : "
        f"{duration:.6f} sec"
    )

    print()

    # --------------------------------------------------------
    # Transition detection
    # --------------------------------------------------------

    selected = detect_transitions(
        frames,
        frame_diff,
        fps,
        total_frames
    )

    print()
    print(
        "最終選出的 5 個轉場："
    )

    cut_frames = []

    for i, item in enumerate(
        selected,
        start=1
    ):

        frame = int(
            item["frame"]
        )

        original_frame = int(
            item.get(
                "original_frame",
                frame
            )
        )

        cut_frames.append(
            frame
        )

        # ----------------------------------------------------
        # 顯示 global → refined
        # ----------------------------------------------------

        if (
            original_frame
            !=
            frame
        ):

            refine_text = (
                f"{original_frame} "
                f"→ {frame}"
            )

        else:

            refine_text = (
                f"{frame} "
                f"(不變)"
            )

        print(
            f"    Transition {i}: "
            f"global/refined = "
            f"{refine_text}, "
            f"time = "
            f"{frame / fps:.6f} sec, "
            f"score = "
            f"{item['transition_score']:.4f}"
        )

        print(
            f"        cross = "
            f"{item['cross_change']:.3f}, "
            f"motion = "
            f"{item['internal_motion']:.3f}, "
            f"ratio = "
            f"{item['stability_ratio']:.3f}"
        )

        if (
            "refine_threshold"
            in item
        ):

            print(
                f"        refine peak = "
                f"{item['refined_peak_frame']}, "
                f"baseline = "
                f"{item['refine_baseline']:.3f}, "
                f"threshold = "
                f"{item['refine_threshold']:.3f}"
            )

        if item.get(
            "refine_reverted",
            False
        ):

            print(
                "        refinement 已退回 global frame"
            )

    # --------------------------------------------------------
    # 確認 5 個切點合法
    # --------------------------------------------------------

    if len(
        cut_frames
    ) != NUM_CUTS:

        raise RuntimeError(
            f"切點數量錯誤："
            f"{len(cut_frames)}"
        )

    if any(
        cut_frames[i]
        >=
        cut_frames[i + 1]

        for i in range(
            len(cut_frames) - 1
        )
    ):

        raise RuntimeError(
            "Transition frame 順序錯誤"
        )

    # --------------------------------------------------------
    # 切影片
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print()
    print(
        "開始切成 6 個 panel："
    )

    cut_video(
        video_path=video_path,
        output_dir=output_dir,
        cut_frames=cut_frames,
        fps=fps,
        total_frames=total_frames
    )

    # --------------------------------------------------------
    # 儲存結果
    # --------------------------------------------------------

    save_transition_info(
        output_dir,
        selected,
        fps,
        total_frames
    )

    if not already_clipped(
        output_dir
    ):

        raise RuntimeError(
            "沒有成功產生完整 6 個 panel"
        )

    print()
    print(
        f"[DONE] {filename}"
    )

    print(
        f"       Output : "
        f"{output_dir}"
    )

    return "success"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "OpenCV 6-panel Global + Local Transition Detection"
    )

    print("=" * 70)

    if not os.path.isdir(
        NEED_CLIP_DIR
    ):

        raise FileNotFoundError(
            f"找不到資料夾："
            f"{NEED_CLIP_DIR}"
        )

    # --------------------------------------------------------
    # 找全部影片
    # --------------------------------------------------------

    video_files = []

    for filename in sorted(
        os.listdir(
            NEED_CLIP_DIR
        )
    ):

        full_path = os.path.join(
            NEED_CLIP_DIR,
            filename
        )

        if not os.path.isfile(
            full_path
        ):
            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in VIDEO_EXTENSIONS:

            video_files.append(
                full_path
            )

    print()
    print(
        f"共找到 "
        f"{len(video_files)} 支影片。"
    )

    if len(video_files) == 0:

        print(
            "need_clip 裡沒有影片。"
        )

        return

    success_count = 0
    skip_count = 0
    fail_count = 0

    failed = []

    # --------------------------------------------------------
    # 逐支處理
    # --------------------------------------------------------

    for video_path in video_files:

        filename = os.path.basename(
            video_path
        )

        try:

            result = process_video(
                video_path
            )

            if result == "success":

                success_count += 1

            elif result == "skip":

                skip_count += 1

        except Exception as e:

            fail_count += 1

            failed.append(
                filename
            )

            print()
            print(
                f"[ERROR] "
                f"{filename}"
            )

            print(
                f"        {e}"
            )

    # --------------------------------------------------------
    # 統計
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "全部處理完成"
    )

    print("=" * 70)

    print(
        f"成功切割   : "
        f"{success_count}"
    )

    print(
        f"已存在跳過 : "
        f"{skip_count}"
    )

    print(
        f"失敗       : "
        f"{fail_count}"
    )

    if failed:

        print()
        print(
            "失敗影片："
        )

        for name in failed:

            print(
                f"    - {name}"
            )


# ============================================================
# 執行
# ============================================================

if __name__ == "__main__":

    main()