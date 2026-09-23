#!/usr/bin/env python3
"""Audit and symlink-stage all StoryMem and ViMax baseline videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ("dualjustice1", "dualjustice2", "elf2", "elf3", "goblin", "kick", "lovehina", "mado")
METHODS = ("storymem", "vimax", "manga2animate")
METHOD_CASES = {
    "storymem": CASES,
    "vimax": CASES,
    "manga2animate": ("dualjustice2",),
}
VIMAX_DIRS = {
    "dualjustice1": "dualjustice1_final_video",
    "dualjustice2": "dualjustice2_final_video",
    "elf2": "elf2_final_video",
    "elf3": "elf3_final_video",
    "goblin": "goblin_final_video",
    "kick": "kick_final_video",
    "lovehina": "lovehina_final_video",
    "mado": "mado_final_video",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def video_path(method: str, case: str, index: int) -> Path:
    if method == "storymem":
        return ROOT / f"manga2animate_video/StoryMem_baseline/{case}/panel_{index:02d}.mp4"
    if method == "vimax":
        folder = VIMAX_DIRS[case]
        filename = f"segment_{index}.mp4" if case == "elf2" else f"panel_{index:02d}.mp4"
        return ROOT / f"manga2animate_video/ViMax_baseline/{folder}/{filename}"
    return ROOT / f"manga2animate_video/dualjustice2_manga2animate/best_video (12)/panel_{index:02d}.mp4"


def character_ref(case: str, index: int) -> Path | None:
    return first_file([
        f"manga2animate_character_ref/{case}/ref_{index}.png",
        f"manga2animate_character_ref/{case}/ref_{index}.jpg",
        f"manga2animate_character_ref/{case}/ref{index}.png",
        f"manga2animate_character_ref/{case}/ref{index}.jpg",
    ])


def scene_ref(case: str, index: int) -> Path | None:
    return first_file([
        f"manga2animate_scene_ref/{case}/panel{index}.png",
        f"manga2animate_scene_ref/{case}/panel{index}.jpg",
        f"manga2animate_scene_ref/{case}/manga_panel{index}.jpg",
    ])


def prompt_path(case: str, index: int) -> Path | None:
    return first_file([
        f"manga2animate_prompt/{case}/panel{index}.txt",
        f"manga2animate_prompt/{case}/panel{index}.txt.txt",
    ])


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"conflicting staging path: {destination}")
        return
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    for name in ("logs", "raw", "summary", "workspace", "audit"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    manifest = []
    required_missing: list[str] = []
    invalid_videos: list[str] = []
    character_missing: list[dict] = []
    for method in METHODS:
        for case in METHOD_CASES[method]:
            for index in range(1, 7):
                video = video_path(method, case, index).resolve()
                character = character_ref(case, index)
                scene = scene_ref(case, index)
                prompt = prompt_path(case, index)
                if not video.is_file():
                    required_missing.append(str(video))
                if scene is None:
                    required_missing.append(f"scene reference: {case} segment {index}")
                if prompt is None:
                    required_missing.append(f"prompt: {case} segment {index}")
                if character is None:
                    character_missing.append({"method": method, "case": case, "segment": index})
                row = {
                    "case": case, "segment": index, "method": method,
                    "video_path": str(video) if video.is_file() else "",
                    "character_ref": str(character) if character else "",
                    "scene_ref": str(scene) if scene else "",
                    "prompt_path": str(prompt) if prompt else "",
                    "video_sha256": sha256(video) if video.is_file() else "",
                    "character_ref_sha256": sha256(character) if character else "",
                    "scene_ref_sha256": sha256(scene) if scene else "",
                    "i2v_subject_status": "APPLICABLE" if character else "SKIP_NO_CHARACTER",
                    "missing_reason": "" if video.is_file() else "source video not present",
                }
                if video.is_file():
                    try:
                        row.update(probe(video))
                    except Exception as error:
                        invalid_videos.append(f"{video}: {type(error).__name__}: {error}")
                manifest.append(row)
                if video.is_file():
                    base = run_dir / "workspace" / f"{case}_{method}"
                    stem = f"segment_{index}"
                    link(video, base / "videos" / f"{stem}.mp4")
                    if scene:
                        link(video, base / "background_videos" / f"{stem}.mp4")
                        link(scene, base / "scene_refs" / f"{stem}{scene.suffix.lower()}")
                    if character:
                        link(video, base / "subject_videos" / f"{stem}.mp4")
                        link(character, base / "character_refs" / f"{stem}{character.suffix.lower()}")

    audit = {
        "cases": list(CASES), "methods": list(METHODS),
        "expected_video_count": sum(len(METHOD_CASES[method]) for method in METHODS) * 6,
        "manifest_row_count": len(manifest),
        "required_missing": sorted(set(required_missing)),
        "invalid_videos": invalid_videos,
        "character_reference_missing": character_missing,
        "character_missing_policy": "skip only original and DINOv3 i2v_subject",
        "vimax_lovehina_directory": "lovehina_final_video",
        "preprocessing": "symlink-only staging; no transcode, resize, crop, or source mutation",
    }
    (run_dir / "audit/input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "audit/dinov3_manifest.json").write_text(
        json.dumps([row for row in manifest if row["character_ref"]], indent=2) + "\n"
    )
    for method in METHODS:
        selected = [row for row in manifest if row["method"] == method]
        (run_dir / f"audit/{method}_manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    (run_dir / "audit/input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if required_missing or invalid_videos:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
