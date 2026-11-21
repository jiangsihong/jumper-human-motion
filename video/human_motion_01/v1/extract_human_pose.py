"""Extract observed human landmarks. No robot control or training is performed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / ".tools/models/pose_landmarker_full.task"
CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 31),
    (28, 32),
]
JOBS = [
    ("01_side_step", "clips/01_side_step.mp4", 2.8),
    ("02_kick_step", "clips/02_kick_step.mp4", 12.4),
    ("03_arm_body_sway", "clips/03_arm_body_sway.mp4", 24.9),
]


def main():
    reports = []
    for name, source, offset in JOBS:
        reader = imageio.get_reader(str(ROOT / source), "ffmpeg")
        meta = reader.get_meta_data()
        fps = meta["fps"]
        out = ROOT / "poses" / name
        out.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(ROOT / "previews" / f"{name}_human_pose.mp4"),
            meta["size"],
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        next(writer)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        points, worlds, vis, presence, detected = [], [], [], [], []
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            for i, frame in enumerate(reader):
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)),
                    round(1000 * i / fps),
                )
                found = bool(result.pose_landmarks)
                detected.append(found)
                if found:
                    p = result.pose_landmarks[0]
                    w = result.pose_world_landmarks[0]
                    xy = np.array([[v.x, v.y, v.z] for v in p], dtype=np.float32)
                    xyz = np.array([[v.x, v.y, v.z] for v in w], dtype=np.float32)
                    v = np.array([v.visibility for v in p], dtype=np.float32)
                    pr = np.array([v.presence for v in p], dtype=np.float32)
                else:
                    xy = xyz = np.full((33, 3), np.nan, dtype=np.float32)
                    v = pr = np.zeros(33, dtype=np.float32)
                points.append(xy)
                worlds.append(xyz)
                vis.append(v)
                presence.append(pr)
                canvas = frame.copy()
                pixels = xy[:, :2] * np.array(meta["size"])
                for a, b in CONNECTIONS:
                    if min(v[a], v[b]) >= 0.5:
                        cv2.line(
                            canvas,
                            tuple(pixels[a].astype(int)),
                            tuple(pixels[b].astype(int)),
                            (40, 255, 80),
                            3,
                        )
                for j in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
                    if v[j] >= 0.5:
                        cv2.circle(canvas, tuple(pixels[j].astype(int)), 5, (255, 100, 50), -1)
                cv2.putText(
                    canvas,
                    f"Observed pose | source {offset + i / fps:.2f}s",
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                writer.send(np.ascontiguousarray(canvas))
        reader.close()
        writer.close()
        points, worlds, vis, presence = map(np.asarray, (points, worlds, vis, presence))
        np.savez_compressed(
            out / "human_landmarks.npz",
            time=np.arange(len(points), dtype=np.float32) / fps,
            source_time=offset + np.arange(len(points), dtype=np.float32) / fps,
            image_landmarks=points,
            world_landmarks=worlds,
            visibility=vis,
            presence=presence,
            detected=np.asarray(detected),
            fps=np.float32(fps),
        )
        joints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        report = dict(
            clip=name,
            source=source,
            frames=len(points),
            detected_frames=int(sum(detected)),
            observed_landmark_mean_visibility=float(np.mean(vis[:, joints])),
            frame_fraction_all_12_landmarks_visible=float(
                np.mean(np.min(vis[:, joints], axis=1) >= 0.5)
            ),
            method="MediaPipe Pose Landmarker VIDEO, 33 landmarks; not motion capture ground truth",
            image_coordinates="normalized x-right y-down; z is model-estimated depth",
            world_coordinates="model-estimated hip-centered metric coordinates; monocular ambiguity remains",
            model_url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            mediapipe_version=mp.__version__,
        )
        (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(name, "detected", sum(detected), "/", len(points), flush=True)
    (ROOT / "poses" / "extraction_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
