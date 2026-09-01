#!/usr/bin/env python3
"""Evaluate VBench++ I2V Subject with the Meta official DINOv3 ViT-B/16 backbone.

This intentionally keeps the published VBench++ I2V Subject scoring formula,
while replacing the DINO v1 feature extractor with the pinned Meta official
DINOv3 constructor and verified checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DINOV3_REPO = ROOT / "third_party/dinov3_official"
CHECKPOINT = DINOV3_REPO / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
sys.path.insert(0, str(DINOV3_REPO))
from dinov3.hub.backbones import dinov3_vitb16  # noqa: E402
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor  # noqa: E402


MODEL_ID = "facebookresearch/dinov3 dinov3_vitb16 LVD-1689M"
SKIPS = {
    ("goblin", 3): "SKIP_INVALID_REFERENCE",
    ("elf2", 1): "SKIP_NO_CHARACTER",
    ("elf2", 6): "SKIP_NO_CHARACTER",
    ("elf3", 3): "SKIP_NO_CHARACTER",
    ("mado", 2): "SKIP_NO_CHARACTER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=("manga2animate", "animaker", "storymem", "vimax"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--case")
    parser.add_argument("--segment", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embed(model, transform, images: list[Image.Image], device: torch.device, batch_size: int) -> torch.Tensor:
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([transform(image) for image in images[start:start + batch_size]]).to(device)
            output = model(batch)
            features.append(F.normalize(output.float(), p=2, dim=-1).cpu())
    return torch.cat(features)


def evaluate_one(model, transform, row: dict, device: torch.device, batch_size: int) -> dict:
    started = time.perf_counter()
    video_path = Path(row["video_path"])
    reference_path = Path(row["character_ref"])
    reference = Image.open(reference_path).convert("RGB")
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    frame_count = len(reader)
    if frame_count < 2:
        raise ValueError(f"at least 2 frames required, found {frame_count}")
    frames_array = reader.get_batch(range(frame_count)).asnumpy()
    frames = [Image.fromarray(frame).convert("RGB") for frame in frames_array]
    reference_feature = embed(model, transform, [reference], device, 1)
    frame_features = embed(model, transform, frames, device, batch_size)
    conformity = torch.clamp(frame_features @ reference_feature.T, min=0).squeeze(1).numpy()
    consecutive = torch.clamp((frame_features[:-1] * frame_features[1:]).sum(dim=1), min=0).numpy()
    conformity_max = float(np.max(conformity))
    consecutive_mean = float(np.mean(consecutive))
    consecutive_min = float(np.min(consecutive))
    score = 0.4 * conformity_max + 0.3 * consecutive_mean + 0.3 * consecutive_min
    return {
        "case": row["case"], "segment": int(row["segment"]), "method": row["method"],
        "video_path": str(video_path), "character_ref": str(reference_path),
        "evaluated_frame_count": frame_count, "feature_dimension": int(frame_features.shape[1]),
        "reference_similarity_max": conformity_max,
        "reference_similarity_mean": float(np.mean(conformity)),
        "consecutive_similarity_mean": consecutive_mean,
        "consecutive_similarity_min": consecutive_min,
        "i2v_subject_dinov3_official": float(score), "status": "SUCCESS", "error": "",
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    model = dinov3_vitb16(pretrained=True, weights=str(CHECKPOINT), check_hash=True).eval().to(device)
    data_config = {"input_size": [3, 256, 256], "interpolation": "bicubic", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    transform = Compose([Resize((256, 256), interpolation=InterpolationMode.BICUBIC), ToTensor(), Normalize(data_config["mean"], data_config["std"])])
    manifest = json.loads(args.manifest.read_text())
    selected = [row for row in manifest if row["method"] == args.method]
    if args.case:
        selected = [row for row in selected if row["case"] == args.case]
    if args.segment is not None:
        selected = [row for row in selected if int(row["segment"]) == args.segment]
    results = []
    for row in selected:
        skip = SKIPS.get((row["case"], int(row["segment"])))
        if skip:
            results.append({
                "case": row["case"], "segment": int(row["segment"]), "method": row["method"],
                "video_path": row["video_path"], "character_ref": row["character_ref"],
                "evaluated_frame_count": 0, "feature_dimension": 768,
                "reference_similarity_max": "", "reference_similarity_mean": "",
                "consecutive_similarity_mean": "", "consecutive_similarity_min": "",
                "i2v_subject_dinov3_official": "", "status": skip, "error": "", "runtime_seconds": 0,
            })
            continue
        if not row.get("video_path") or not row.get("character_ref"):
            raise FileNotFoundError(f"missing applicable input: {row['case']} segment {row['segment']} {row['method']}")
        results.append(evaluate_one(model, transform, row, device, args.batch_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    audit = {
        "model_id": MODEL_ID,
        "implementation": "Meta official constructor at commit 6876159a11b4df116f30f667f8c9888617df0751",
        "checkpoint_path": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "model_data_config": data_config,
        "vbench_formula": "0.4*max(reference_similarity)+0.3*mean(consecutive_similarity)+0.3*min(consecutive_similarity)",
        "input_manifest": str(args.manifest.resolve()),
        "method": args.method, "physical_gpu": args.gpu, "result_count": len(results),
        "success_count": sum(row["status"] == "SUCCESS" for row in results),
        "skip_count": sum(row["status"].startswith("SKIP_") for row in results),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
