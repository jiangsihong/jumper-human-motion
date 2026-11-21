"""Extract visually reviewed action intervals; no pose estimation or training.

Use the existing pytorch environment. Frame intervals are zero-based,
start-inclusive and end-exclusive at 30 fps. The source stays unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / "b667ffa51c09b5d564d96c248cec54f4.mp4"
FPS = 30
CLIPS = [
    dict(
        id="01_side_step",
        file="clips/01_side_step.mp4",
        frames=[84, 279],
        description="Repeated lateral evasive steps with arm and torso movement.",
        full_body=True,
    ),
    dict(
        id="02_kick_step",
        file="clips/02_kick_step.mp4",
        frames=[372, 495],
        description="Crossing/kicking leg movement with both arms extended.",
        full_body=True,
    ),
    dict(
        id="03_arm_body_sway",
        file="clips/03_arm_body_sway.mp4",
        frames=[747, 969],
        description="Repeated arm opening/lifting, body sway and stepping.",
        full_body=False,
        note="Source cuts to close views around 28.37--30.30 s; feet are partly hidden.",
    ),
    dict(
        id="03_full_body_excerpt",
        file="clips/full_body/03_arm_body_sway_full_body.mp4",
        frames=[747, 850],
        description="Continuous full-body excerpt before the first close-view edit.",
        full_body=True,
        note="An additional partial excerpt, not the complete third performance.",
    ),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_sheet(path: Path, destination: Path) -> None:
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        count = reader.count_frames()
        indexes = [round((count - 1) * i / 11) for i in range(12)]
        sheet = Image.new("RGB", (1200, 3 * 238), "white")
        draw = ImageDraw.Draw(sheet)
        for i, index in enumerate(indexes):
            frame = Image.fromarray(reader.get_data(index))
            frame.thumbnail((300, 211))
            x, y = i % 4 * 300, i // 4 * 238
            sheet.paste(frame, (x + (300 - frame.width) // 2, y + 24))
            draw.text((x + 5, y + 5), f"clip {index / FPS:.2f}s", fill="black")
        sheet.save(destination, quality=92)
    finally:
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="user-supplied source video; raw footage is not distributed",
    )
    source = parser.parse_args().source.expanduser().resolve()
    if not source.is_file():
        parser.error("Source video is not distributed; pass --source /path/to/source.mp4")
    source_hash = sha256(source)
    original = imageio.get_reader(str(source), "ffmpeg")
    metadata = original.get_meta_data()
    original.close()
    assert metadata["fps"] == FPS, metadata
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    results = []
    for entry in CLIPS:
        target = ROOT / entry["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        start, end = entry["frames"]
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
                "-map",
                "0:v:0",
                "-vf",
                f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS",
                "-an",
                "-r",
                str(FPS),
                "-fps_mode",
                "cfr",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(target),
            ],
            check=True,
        )
        reader = imageio.get_reader(str(target), "ffmpeg")
        assert reader.get_meta_data()["fps"] == FPS
        frames = 0
        size = None
        for frame in reader:
            frames += 1
            size = list(frame.shape[1::-1])
        reader.close()
        assert frames == end - start, (entry["id"], frames, end - start)
        assert size == list(metadata["size"]), size
        results.append(
            dict(
                entry,
                source_start_s=start / FPS,
                source_end_s=end / FPS,
                duration_s=frames / FPS,
                frame_count=frames,
                fps=FPS,
                size=size,
                audio=False,
                sha256=sha256(target),
                validation="all frames decoded; exact frame count and size checked",
            )
        )
        contact_sheet(target, ROOT / "previews" / f"{entry['id']}.jpg")
        print(entry["file"], frames, f"{frames / FPS:.3f}s", flush=True)
    assert sha256(source) == source_hash, "Source video changed"
    manifest = dict(
        source=str(source),
        source_sha256=source_hash,
        source_metadata={key: metadata[key] for key in ("fps", "duration", "size")},
        method="Manual visual selection; frame-accurate re-encoding.",
        scope="Video extraction only. No human keypoints, robot trajectories, or training.",
        audio_policy="Silent clips; original audio remains in the unchanged source.",
        clips=results,
    )
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
