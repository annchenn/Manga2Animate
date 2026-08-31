#!/usr/bin/env python3
"""Audit six-case inputs and create an isolated, symlink-only evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ("dualjustice1", "elf2", "elf3", "goblin", "mado", "lovehina")
METHODS = ("manga2animate", "animaker")

VIDEO_LAYOUTS = {
    ("dualjustice1", "manga2animate"): "manga2animate_video/dualjustice1_manga2animate/best_video_6/segment_{index}.mp4",
    ("dualjustice1", "animaker"): "manga2animate_video/dualjustice1_animaker/dualjustice1/segment_{index}.mp4",
    ("elf2", "manga2animate"): "manga2animate_video/elf2_manga2animate/best_video/segment_{index}.mp4",
    ("elf2", "animaker"): "manga2animate_video/elf2_animaker/elf2/segment_{index}.mp4",
    ("elf3", "manga2animate"): "manga2animate_video/elf3_manga2animate/elf3_segments/shot_{index:02d}.mp4",
    ("elf3", "animaker"): "manga2animate_video/elf3_animaker/elf3_segments/shot_{index:02d}.mp4",
    ("goblin", "manga2animate"): "manga2animate_video/goblin_manga2animate/goblin_segments/shot_{index:02d}.mp4",
    ("goblin", "animaker"): "manga2animate_video/goblin_animaker/goblin_segments/shot_{index:02d}.mp4",
    ("mado", "manga2animate"): "manga2animate_video/mado_manga2animate_new/split_segments/segment_{index}.mp4",
    ("mado", "animaker"): "manga2animate_video/mado_animaker/mado_animaker/segment_{index}.mp4",
    ("lovehina", "manga2animate"): "manga2animate_video/lovehina_manga2animate/best_video_9_segments/segment_{index}.mp4",
    ("lovehina", "animaker"): "manga2animate_video/lovehina_animaker/LoveHina_segments/segment_{index}.mp4",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def first_existing(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        path = ROOT / pattern
        if path.is_file():
            return path.resolve()
    return None


def reference(case: str, index: int, kind: str) -> Path | None:
    if kind == "character":
        patterns = [
            f"manga2animate_character_ref/{case}/ref_{index}.png",
            f"manga2animate_character_ref/{case}/ref_{index}.jpg",
            f"manga2animate_character_ref/{case}/ref{index}.jpg",
        ]
    else:
        patterns = [
            f"manga2animate_scene_ref/{case}/panel{index}.png",
            f"manga2animate_scene_ref/{case}/panel{index}.jpg",
            f"manga2animate_scene_ref/{case}/manga_panel{index}.jpg",
        ]
    return first_existing(patterns)


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames",
        "-show_entries", "format=duration,size", "-of", "json", str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    stream = payload["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    return {
        "codec": stream["codec_name"], "width": int(stream["width"]),
        "height": int(stream["height"]), "pix_fmt": stream["pix_fmt"],
        "fps_original": numerator / denominator,
        "source_frame_count": int(stream["nb_read_frames"]),
        "duration_seconds": float(payload["format"]["duration"]),
        "size_bytes": int(payload["format"]["size"]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(target: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != target.resolve():
            raise RuntimeError(f"conflicting staging link: {destination}")
        return
    destination.symlink_to(target)


def main() -> None:
    run_dir = args().run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "raw", "summary", "workspace", "audit"):
        (run_dir / name).mkdir(exist_ok=True)
    manifest = []
    for case in CASES:
        for method in METHODS:
            layout = VIDEO_LAYOUTS.get((case, method))
            for index in range(1, 7):
                video = (ROOT / layout.format(index=index)).resolve() if layout else None
                if video is not None and not video.is_file():
                    video = None
                character = reference(case, index, "character")
                scene = reference(case, index, "scene")
                row = {
                    "case": case, "segment": index, "method": method,
                    "video_path": str(video) if video else "",
                    "character_ref": str(character) if character else "",
                    "scene_ref": str(scene) if scene else "",
                    "video_sha256": sha256(video) if video else "",
                    "character_ref_sha256": sha256(character) if character else "",
                    "scene_ref_sha256": sha256(scene) if scene else "",
                    "missing_reason": "" if video else "source video not present in workspace",
                }
                if video:
                    row.update(probe(video))
                manifest.append(row)
                if video:
                    stem = f"segment_{index}"
                    base = run_dir / "workspace" / f"{case}_{method}"
                    link(video, base / "videos" / f"{stem}.mp4")
                    if character:
                        link(character, base / "character_refs" / f"{stem}{character.suffix.lower()}")
                        link(video, base / "subject_videos" / f"{stem}.mp4")
                    if scene:
                        link(scene, base / "scene_refs" / f"{stem}{scene.suffix.lower()}")
                        link(video, base / "background_videos" / f"{stem}.mp4")
    (run_dir / "audit" / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = {
        "created_at": subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip(),
        "repository_root": str(ROOT), "run_dir": str(run_dir),
        "preprocessing": "symlink-only staging; no transcode, resize, crop, or source mutation",
        "dynamic_frame_sampling": "interval=max(1, round(source_fps/8)); frames[::interval]",
        "normalization": "continuous values divided by min(width,height)/256",
    }
    (run_dir / "audit" / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    present = sum(bool(row["video_path"]) for row in manifest)
    print(f"Created {run_dir}; {present}/{len(manifest)} anticipated segment videos present")


if __name__ == "__main__":
    main()
