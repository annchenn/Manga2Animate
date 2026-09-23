#!/usr/bin/env python3
"""Continuous Dynamic Degree using VBench's official RAFT configuration.

This evaluator deliberately mirrors ``vbench.dynamic_degree.DynamicDegree``:
RAFT-Things, 20 update iterations, and frame sampling at approximately 8 fps.
Unlike the official Boolean metric, it preserves continuous flow statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from easydict import EasyDict

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


FIELDS = [
    "case", "segment", "method", "video_path", "fps_original",
    "fps_evaluated", "width", "height", "evaluated_frame_count",
    "flow_pair_count", "modified_dd_mean", "spatial_std_mean",
    "temporal_mean_flow_std", "mean_top5_flow",
    "normalized_modified_dd_mean", "normalized_spatial_std_mean",
    "success", "error", "runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", help="only the manifest rows with this method name (default: all)")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--alternate-corr", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device, alternate_corr: bool) -> RAFT:
    args = EasyDict({
        "model": str(checkpoint),
        "small": False,
        "mixed_precision": False,
        "alternate_corr": alternate_corr,
    })
    model = RAFT(args)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict({key.replace("module.", ""): value for key, value in payload.items()})
    return model.to(device).eval()


def inspect_video(path: Path) -> tuple[float, int, int, list[torch.Tensor], int]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    interval = max(1, round(fps / 8.0))
    frames: list[torch.Tensor] = []
    source_index = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1).float()[None])
        source_index += 1
    capture.release()
    if not frames:
        raise RuntimeError("video decoded zero frames")
    return fps, width, height, frames, interval


def evaluate_video(model: RAFT, device: torch.device, path: Path) -> dict[str, float | int]:
    fps, width, height, cpu_frames, interval = inspect_video(path)
    mean_flows: list[float] = []
    spatial_stds: list[float] = []
    top5_flows: list[float] = []
    with torch.inference_mode():
        for cpu_image1, cpu_image2 in zip(cpu_frames[:-1], cpu_frames[1:]):
            image1 = cpu_image1.to(device)
            image2 = cpu_image2.to(device)
            padder = InputPadder(image1.shape)
            padded1, padded2 = padder.pad(image1, image2)
            _, flow = model(padded1, padded2, iters=20, test_mode=True)
            flow = padder.unpad(flow)[0]
            magnitude = torch.sqrt(flow[0].square() + flow[1].square()).float()
            flat = magnitude.flatten()
            top_count = max(1, int(flat.numel() * 0.05))
            mean_flows.append(float(flat.mean().item()))
            spatial_stds.append(float(flat.std(unbiased=False).item()))
            top5_flows.append(float(torch.topk(flat, top_count, sorted=False).values.mean().item()))
            del image1, image2, padded1, padded2, flow, magnitude, flat
    if not mean_flows:
        raise RuntimeError("fewer than two evaluated frames")
    scale = min(width, height) / 256.0
    return {
        "fps_original": fps,
        "fps_evaluated": fps / interval,
        "width": width,
        "height": height,
        "evaluated_frame_count": len(cpu_frames),
        "flow_pair_count": len(mean_flows),
        "modified_dd_mean": float(np.mean(mean_flows)),
        "spatial_std_mean": float(np.mean(spatial_stds)),
        "temporal_mean_flow_std": float(np.std(mean_flows, ddof=0)),
        "mean_top5_flow": float(np.mean(top5_flows)),
        "normalized_modified_dd_mean": float(np.mean(mean_flows) / scale),
        "normalized_spatial_std_mean": float(np.mean(spatial_stds) / scale),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = [
        row for row in manifest
        if (args.method is None or row["method"] == args.method)
        and (not args.cases or row["case"] in args.cases)
    ]
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, args.alternate_corr)
    results: list[dict] = []
    for item in selected:
        started = time.perf_counter()
        base = {key: item.get(key, "") for key in ("case", "segment", "method", "video_path")}
        video_path = Path(item["video_path"]) if item.get("video_path") else None
        try:
            if video_path is None or not video_path.is_file():
                raise FileNotFoundError(item.get("missing_reason") or "source video missing")
            values = evaluate_video(model, device, video_path)
            row = {**base, **values, "success": True, "error": ""}
        except Exception as error:  # preserve failures as auditable rows
            row = {**base, "success": False, "error": f"{type(error).__name__}: {error}"}
        row["runtime_seconds"] = time.perf_counter() - started
        for field in FIELDS:
            row.setdefault(field, "")
        results.append(row)
        write_rows(args.output, results)
        print(json.dumps({key: row[key] for key in ("case", "segment", "method", "success", "error", "runtime_seconds")}))


if __name__ == "__main__":
    main()
