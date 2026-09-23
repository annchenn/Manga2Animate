import os
import cv2
import numpy as np
from itertools import combinations

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ============================================================
# Settings
# ============================================================

NEED_CLIP_DIR = "need_clip"

# Fixed 6 panels -> 5 transitions
NUM_PANELS = 6
NUM_CUTS = NUM_PANELS - 1

# ------------------------------------------------------------
# True：
#   re-cut every video, even if its folder already has all 6 panels
#
# False：
#   skip a video whose folder already has panel_01 ~ panel_06
# ------------------------------------------------------------

FORCE_RECUT = False


# ============================================================
# Global candidate peak settings
# ============================================================

# minimum spacing between candidate peaks
CANDIDATE_DISTANCE_SECONDS = 0.20

# minimum spacing between any two final transitions
MIN_PANEL_SECONDS = 0.50

# transitions may not be this close to the start or end of the video
EDGE_SECONDS = 0.40

# Global frame difference smoothing
SMOOTH_SIGMA = 1.5

# at most this many candidate peaks go into the combination search
MAX_CANDIDATES = 30


# ============================================================
# Persistent transition analysis
# ============================================================

# seconds analysed before and after each global candidate
TRANSITION_WINDOW_SECONDS = 0.35

# frames are downscaled to this width to speed up analysis
ANALYSIS_WIDTH = 320


# ============================================================
# Global transition score weights
# ============================================================

# single-frame change
WEIGHT_LOCAL_DIFF = 0.15

# peak prominence
WEIGHT_PROMINENCE = 0.20

# difference between the mean frames before and after the transition
WEIGHT_CROSS_CHANGE = 0.40

# before/after difference relative to the motion inside each side
WEIGHT_STABILITY_RATIO = 0.25


# ============================================================
# Local boundary refinement settings
# ============================================================

# seconds searched backwards from the global transition
REFINE_LOOKBACK_SECONDS = 0.40

# seconds still searched after the global transition
REFINE_LOOKAHEAD_SECONDS = 0.12

# earlier region used to estimate the normal motion baseline
REFINE_BASELINE_SECONDS = 0.35

# fraction of the transition peak used as the onset threshold
#
# smaller:
#   pulls the boundary earlier more easily
#
# larger:
#   stays closer to the transition peak
REFINE_PEAK_FRACTION = 0.25

# baseline + MAD * multiplier
REFINE_MAD_MULTIPLIER = 2.5

# when searching backwards, this many consecutive frames below the
# threshold mark the end of the transition
REFINE_QUIET_FRAMES = 2

# lighter smoothing used for refinement
REFINE_SMOOTH_SIGMA = 0.5


# ============================================================
# Supported formats
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
# Check whether a video has already been fully cut
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
# Remove old output
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
                    f"Warning: could not delete {path}"
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
# Read the video + frame differences
# ============================================================

def load_video_analysis(video_path):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video: {video_path}"
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
            f"Cannot read FPS: {video_path}"
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
            "Not enough frames in the video"
        )

    total_frames = len(frames)

    if metadata_frames != total_frames:

        print(
            f"Note: metadata frames = "
            f"{metadata_frames}, "
            f"actually read = {total_frames}"
        )

    # --------------------------------------------------------
    # differences between adjacent frames
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
# Find candidate peaks
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
# Compute candidate transition features
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
        # mean frame before the transition
        # ----------------------------------------------------

        before_mean = np.mean(
            np.stack(
                before_frames,
                axis=0
            ),
            axis=0
        )

        # ----------------------------------------------------
        # mean frame after the transition
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
        # motion inside the segment before the transition
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
        # motion inside the segment after the transition
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
# Pick the best 5 global transitions from the candidates
# ============================================================

def select_best_five(
    candidates,
    total_frames,
    fps
):

    if len(candidates) < NUM_CUTS:

        raise RuntimeError(
            f"only "
            f"{len(candidates)} candidates, "
            f"fewer than {NUM_CUTS}."
        )

    # --------------------------------------------------------
    # keep the candidates with the best transition scores first
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

    # sort again by time
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
    # try every 5-cut combination
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

        # reject the combination if any panel is too short
        if np.any(
            segment_lengths
            <
            min_panel_frames
        ):
            continue

        # sum of the five transition scores
        score = sum(
            x["transition_score"]
            for x in combo
        )

        # ----------------------------------------------------
        # avoid extremely long panels
        #
        # penalise slightly only when one panel exceeds 45% of the whole video;
        # this does not force six equal parts
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
            "Could not find 5 valid global transitions"
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
    # refinement uses lighter smoothing
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
    # refine each global transition separately
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
        # the baseline uses the region before the search window
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

        # still no data for the baseline
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
        # largest change inside the local window
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
        # search backwards from the local peak for the transition onset
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
    # order by global transition
    # --------------------------------------------------------

    refined_items = sorted(
        refined_items,
        key=lambda x:
        x["original_frame"]
    )

    # --------------------------------------------------------
    # Safety check
    #
    # if refinement makes a panel too short,
    # that transition falls back to its global frame
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

    # check several times, since one fallback can affect its neighbours
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
    # finally make sure the frames are increasing
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

        # if anything is wrong, fall back to the global transitions for all
        print(
            "Warning: refinement put the cut points out of order; "
            "falling back to the global transitions for all."
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
# Full transition detection
# ============================================================

def detect_transitions(
    frames,
    frame_diff,
    fps,
    total_frames
):

    # --------------------------------------------------------
    # Step 1: find many candidate peaks
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
        f"Initial candidate peaks: "
        f"{len(peaks)}"
    )

    # --------------------------------------------------------
    # Step 2: analyse the frames before and after each candidate
    # --------------------------------------------------------

    features = calculate_candidate_features(
        frames,
        smooth_diff,
        peaks,
        prominences,
        fps
    )

    # --------------------------------------------------------
    # Step 3: compute transition scores
    # --------------------------------------------------------

    features = calculate_transition_scores(
        features
    )

    print(
        f"Valid transition candidates: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Step 4: global search for the best 5
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
# Cut the video with OpenCV
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
            f"Cannot open video: {video_path}"
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
                f"Cannot create: {output_path}"
            )

        # ----------------------------------------------------
        # seek to the panel's first frame
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
                f"No frames written"
            )

        if not os.path.isfile(
            output_path
        ):

            cap.release()

            raise RuntimeError(
                f"Not produced: {output_path}"
            )

        if os.path.getsize(
            output_path
        ) == 0:

            cap.release()

            raise RuntimeError(
                f"Output video is empty: {output_path}"
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
# Save transition info
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
# Process one video
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
    # already fully cut
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
            "       already cut into 6 panels"
        )

        return "skip"

    # --------------------------------------------------------
    # output already exists
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
    # analyse the video
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
        "Final 5 transitions:"
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
        # show global -> refined
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
                f"(unchanged)"
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
                "        refinement fell back to the global frame"
            )

    # --------------------------------------------------------
    # check that the 5 cut points are valid
    # --------------------------------------------------------

    if len(
        cut_frames
    ) != NUM_CUTS:

        raise RuntimeError(
            f"Wrong number of cut points: "
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
            "Transition frames out of order"
        )

    # --------------------------------------------------------
    # cut the video
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print()
    print(
        "Cutting into 6 panels:"
    )

    cut_video(
        video_path=video_path,
        output_dir=output_dir,
        cut_frames=cut_frames,
        fps=fps,
        total_frames=total_frames
    )

    # --------------------------------------------------------
    # save the results
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
            "Did not produce all 6 panels"
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
            f"Folder not found: "
            f"{NEED_CLIP_DIR}"
        )

    # --------------------------------------------------------
    # find all videos
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
        f"Found "
        f"{len(video_files)} video(s)."
    )

    if len(video_files) == 0:

        print(
            "No videos in need_clip."
        )

        return

    success_count = 0
    skip_count = 0
    fail_count = 0

    failed = []

    # --------------------------------------------------------
    # process one by one
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
    # summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "All done"
    )

    print("=" * 70)

    print(
        f"Cut        : "
        f"{success_count}"
    )

    print(
        f"Skipped    : "
        f"{skip_count}"
    )

    print(
        f"Failed     : "
        f"{fail_count}"
    )

    if failed:

        print()
        print(
            "Failed videos:"
        )

        for name in failed:

            print(
                f"    - {name}"
            )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()