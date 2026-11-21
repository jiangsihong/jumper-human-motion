"""Compare the observed human with expressive targets and honest PD executions."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jumper_mapping_renderer import render_reference

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "v1"
JOBS = [("01_side_step", 195), ("02_kick_step", 123), ("03_arm_body_sway", 103)]


def compose(name, count, time_scale, robot_video, mode):
    human = SOURCE / "clips" / f"{name}.mp4"
    if not human.is_file():
        raise SystemExit("Source clips are not distributed; run v1/extract_clips.py --source first")
    label = ROOT / "previews" / f"{name}_{mode}_header.png"
    banner = Image.new("RGB", (1280, 36), "#18202a")
    draw = ImageDraw.Draw(banner)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    draw.text((12, 5), f"Human reference - slowed {time_scale:g}x", font=font, fill="white")
    text = (
        "V2 MAPPING TARGET - NO TRAINING"
        if mode == "target"
        else "V2 FREE PD PHYSICS - NO TRAINING"
    )
    draw.text((652, 5), text, font=font, fill="white")
    banner.save(label)
    suffix = "comparison" if mode == "target" else "physics_comparison"
    output = ROOT / "previews" / f"{name}_{suffix}.mp4"
    filters = (
        f"[0:v]trim=end_frame={count},setpts={time_scale}*(PTS-STARTPTS),fps=30,"
        "tpad=start_mode=clone:start_duration=0.5:stop_mode=clone:stop_duration=0.5,"
        "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2[h];"
        "[1:v]scale=640:480[r];[h][r]hstack=inputs=2:shortest=1[base];"
        "[base][2:v]overlay=0:0:shortest=1[v]"
    )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(human),
            "-i",
            str(robot_video),
            "-loop",
            "1",
            "-i",
            str(label),
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-an",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    reader = imageio.get_reader(str(output), "ffmpeg")
    frames = sum(1 for _ in reader)
    meta = reader.get_meta_data()
    reader.close()
    return dict(path=str(output), frames=frames, fps=meta["fps"], size=meta["size"])


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "mapping/robot_preview.xml"))
    reports = []
    for name, count in JOBS:
        with np.load(ROOT / "mapping" / f"{name}_robot_trajectory.npz") as z:
            target = z["qpos"]
            dt = float(z["dt"])
            time_scale = float(z["time_scale"])
        target_video = ROOT / "previews" / f"{name}_mapped.mp4"
        render_reference(
            model,
            target,
            dt,
            target_video,
            preview_type="V2 expressive mapping target - prescribed poses - no training",
        )
        comparison = compose(name, count, time_scale, target_video, "target")
        with np.load(ROOT / "mapping" / f"{name}_pd_rollout.npz") as z:
            actual = z["qpos"]
        physics_video = ROOT / "previews" / f"{name}_pd_physics.mp4"
        render_reference(
            model,
            actual,
            dt,
            physics_video,
            preview_type="V2 free-base PD physics playback - no root assistance - no training",
        )
        physical_comparison = compose(name, count, time_scale, physics_video, "physics")
        report = dict(
            name=name,
            comparison=comparison,
            physics_comparison=physical_comparison,
            target_is_prescribed=True,
            physics_has_root_assistance=False,
            training_started=False,
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    (ROOT / "previews/preview_manifest.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
