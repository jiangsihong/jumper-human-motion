"""Render recorded PD rollouts and compose observed-video comparisons."""

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
JOBS = [("01_side_step", 195), ("02_kick_step", 123), ("03_arm_body_sway", 103)]


def main():
    if not (ROOT / "clips/01_side_step.mp4").is_file():
        raise SystemExit("Source clips are not distributed; run extract_clips.py --source first")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "mapping/robot_preview.xml"))
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    header = ROOT / "previews" / "comparison_header.png"
    banner = Image.new("RGB", (1280, 36), "#18202a")
    draw = ImageDraw.Draw(banner)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    draw.text((12, 5), "Human reference - slowed 1.8x", font=font, fill="white")
    draw.text((652, 5), "Robot mapping - PD physics - no training", font=font, fill="white")
    banner.save(header)
    results = []
    for name, count in JOBS:
        with np.load(ROOT / "mapping" / f"{name}_pd_rollout.npz") as z:
            physical = z["qpos"]
            dt = float(z["dt"])
        physics_video = ROOT / "previews" / f"{name}_pd_physics.mp4"
        render_reference(
            model,
            physical,
            dt,
            physics_video,
            preview_type="Recorded free-base PD physics playback (no learning)",
        )
        source = ROOT / "clips" / f"{name}.mp4"
        comparison = ROOT / "previews" / f"{name}_comparison.mp4"
        filters = (
            f"[0:v]trim=end_frame={count},setpts=1.8*(PTS-STARTPTS),fps=30,"
            "tpad=start_mode=clone:start_duration=1:stop_mode=clone:stop_duration=1.5,"
            "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2[h];"
            "[1:v]scale=640:480[r];[h][r]hstack=inputs=2:shortest=1[base];"
            "[base][2:v]overlay=0:0:shortest=1[v]"
        )
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-i",
                str(physics_video),
                "-loop",
                "1",
                "-i",
                str(header),
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
                str(comparison),
            ],
            check=True,
        )
        reader = imageio.get_reader(str(comparison), "ffmpeg")
        frames = sum(1 for _ in reader)
        meta = reader.get_meta_data()
        reader.close()
        result = dict(
            name=name,
            comparison=str(comparison),
            frames=frames,
            fps=meta["fps"],
            size=meta["size"],
            physical_video=str(physics_video),
            note="left is retimed observed video; right is untrained simple-PD free-base tracking",
        )
        results.append(result)
        print(result, flush=True)
    (ROOT / "previews" / "preview_manifest.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
