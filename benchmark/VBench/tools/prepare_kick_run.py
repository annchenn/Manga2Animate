#!/usr/bin/env python3
"""Audit and stage the two six-segment kick results without modifying sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("manga2animate", "animaker")
VIDEO_LAYOUTS = {
    "manga2animate": "manga2animate_video/kick_manga2animate/best_video_split/shot_{index:02d}.mp4",
    "animaker": "manga2animate_video/kick_animaker/kick_split/shot_{index:02d}.mp4",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


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


def first_file(patterns: list[str]) -> Path | None:
    for pattern in patterns:
        candidate = ROOT / pattern
        if candidate.is_file():
            return candidate.resolve()
    return None


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source:
            raise RuntimeError(f"conflicting staging path: {destination}")
        return
    destination.symlink_to(source)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    for name in ("logs", "raw", "summary", "workspace", "audit"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    manifest = []
    required_missing: list[str] = []
    character_missing: list[str] = []
    for method in METHODS:
        for index in range(1, 7):
            video = (ROOT / VIDEO_LAYOUTS[method].format(index=index)).resolve()
            scene = first_file([
                f"manga2animate_scene_ref/kick/manga_panel{index}.jpg",
                f"manga2animate_scene_ref/kick/panel{index}.png",
                f"manga2animate_scene_ref/kick/panel{index}.jpg",
            ])
            character = first_file([
                f"manga2animate_character_ref/kick/ref_{index}.png",
                f"manga2animate_character_ref/kick/ref_{index}.jpg",
                f"manga2animate_character_ref/kick/ref{index}.jpg",
            ])
            prompt = ROOT / f"manga2animate_prompt/kick/panel{index}.txt"
            if not video.is_file():
                required_missing.append(str(video))
            if scene is None:
                required_missing.append(f"scene reference for kick segment {index}")
            if not prompt.is_file():
                required_missing.append(str(prompt))
            if character is None:
                character_missing.append(f"kick segment {index} ({method})")
            row = {
                "case": "kick", "segment": index, "method": method,
                "video_path": str(video) if video.is_file() else "",
                "character_ref": str(character) if character else "",
                "scene_ref": str(scene) if scene else "",
                "prompt_path": str(prompt.resolve()) if prompt.is_file() else "",
                "video_sha256": digest(video) if video.is_file() else "",
                "character_ref_sha256": digest(character) if character else "",
                "scene_ref_sha256": digest(scene) if scene else "",
                "i2v_subject_status": "APPLICABLE" if character else "SKIP_NO_CHARACTER",
                "missing_reason": "" if video.is_file() else "source video not present",
            }
            if video.is_file():
                row.update(probe(video))
            manifest.append(row)
            if video.is_file():
                base = run_dir / "workspace" / f"kick_{method}"
                stem = f"segment_{index}"
                link(video, base / "videos" / f"{stem}.mp4")
                if scene:
                    link(video, base / "background_videos" / f"{stem}.mp4")
                    link(scene, base / "scene_refs" / f"{stem}{scene.suffix.lower()}")
                if character:
                    link(video, base / "subject_videos" / f"{stem}.mp4")
                    link(character, base / "character_refs" / f"{stem}{character.suffix.lower()}")

    audit = {
        "case": "kick", "methods": list(METHODS), "segments_per_method": 6,
        "required_missing": sorted(set(required_missing)),
        "character_reference_missing": sorted(set(character_missing)),
        "character_missing_policy": "skip only i2v_subject and DINOv3 i2v_subject",
        "preprocessing": "symlink-only staging; no transcode, resize, crop, or source mutation",
    }
    (run_dir / "audit/input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    applicable = [row for row in manifest if row["character_ref"]]
    (run_dir / "audit/dinov3_manifest.json").write_text(json.dumps(applicable, indent=2) + "\n")
    (run_dir / "audit/input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if required_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
