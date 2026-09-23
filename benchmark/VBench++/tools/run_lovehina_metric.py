#!/usr/bin/env python3
"""Run one isolated lovehina VBench++ metric with execution/resource audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--method", choices=("manga2animate", "animaker"), required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    return parser.parse_args()


def gpu_memory(gpu: int) -> int | None:
    try:
        output = subprocess.check_output([
            "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL)
        return int(output.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    workspace = run_dir / "workspace" / f"{args.case}_{args.method}"
    output = run_dir / "raw" / f"{args.case}_{args.method}" / args.metric
    output.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / f"{args.case}_{args.method}_{args.metric}_gpu{args.gpu}.log"
    audit_path = run_dir / "audit" / f"{args.case}_{args.method}_{args.metric}.json"
    reference = workspace / ("character_refs" if args.metric == "i2v_subject" else "scene_refs")
    entrypoint = ROOT / ("tools/evaluate_i2v_altcorr.py" if args.metric == "dynamic_degree" else "evaluate_i2v.py")
    video_folder = "subject_videos" if args.metric == "i2v_subject" else ("background_videos" if args.metric == "i2v_background" else "videos")
    command = [
        "python", str(entrypoint), "--output_path", str(output),
        "--videos_path", str(workspace / video_folder), "--dimension", args.metric,
        "--mode", "custom_input", "--ratio", "16-9",
    ]
    if args.metric in ("i2v_subject", "i2v_background"):
        command.extend(["--custom_image_folder", str(reference)])
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "VBENCH_CACHE_DIR": str(ROOT / ".cache/vbench"),
        "HF_HOME": str(ROOT / ".cache/huggingface"),
        "TORCH_HOME": str(ROOT / ".cache/torch"),
        "PYTHONPATH": os.pathsep.join([
            str(ROOT / "vbench/third_party/RAFT/alt_cuda_corr"), str(ROOT),
            environment.get("PYTHONPATH", ""),
        ]),
    })
    samples: list[dict] = []
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(1.0):
            samples.append({"seconds": time.perf_counter() - started, "gpu_memory_mib": gpu_memory(args.gpu)})

    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        returncode = process.wait()
        stop.set()
        thread.join()
    runtime = time.perf_counter() - started
    memory_values = [sample["gpu_memory_mib"] for sample in samples if sample["gpu_memory_mib"] is not None]
    audit = {
        "case": args.case, "method": args.method, "metric": args.metric,
        "status": "success" if returncode == 0 else "failed", "returncode": returncode,
        "started_at": started_at, "finished_at": datetime.now().astimezone().isoformat(),
        "runtime_seconds": runtime, "physical_gpu": args.gpu,
        "peak_gpu_memory_mib": max(memory_values) if memory_values else None,
        "command": command, "log_path": str(log_path), "output_path": str(output),
        "resource_samples": samples,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("method", "metric", "status", "runtime_seconds", "peak_gpu_memory_mib", "log_path", "output_path")}))
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
