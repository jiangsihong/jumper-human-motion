"""Render prescribed MuJoCo qpos through Jumper's existing ReplayVideo recorder.

This is a kinematic-reference preview only: it never integrates physics, trains a
policy, or claims that a reference can be executed dynamically. No repository
files are changed. Set MUJOCO_GL=egl before starting Python on headless Linux.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import math

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw
from mjlab.viewer.viewer_config import ViewerConfig
from mjrl.viewer.video import ReplayVideo


def _make_facade(model, qpos, dt, width, height):
    """The smallest single-world state facade consumed by ReplayVideo."""
    state = mujoco.MjData(model)
    state.qpos[:] = qpos[0]
    # The first root pose frames a fixed camera, so apparent sway comes from the
    # reference rather than camera tracking. Jumper faces +X.
    target = tuple(float(v) for v in qpos[0, :3]) if model.nq >= 7 else tuple(model.stat.center)
    lookat = (target[0], target[1], target[2] + 0.015)
    cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.WORLD,
        width=width,
        height=height,
        lookat=lookat,
        distance=0.75,
        azimuth=145.0,
        elevation=-25.0,
        geom_group=(1, 0, 1, 0, 0, 0),
        site_group=(0, 0, 0, 0, 0, 0),
        max_extra_envs=0,
    )
    batched = SimpleNamespace(nworld=1)
    for name in (
        "qpos",
        "qvel",
        "mocap_pos",
        "mocap_quat",
        "ctrl",
        "act",
        "qfrc_applied",
        "xfrc_applied",
    ):
        array = np.array(getattr(state, name), dtype=np.float64, copy=True)
        setattr(batched, name, torch.from_numpy(array[None]))
    batched.time = torch.zeros(1, dtype=torch.float64)
    sim = SimpleNamespace(
        mj_model=model, model=None, expanded_fields=set(), data=batched, visuals_stripped=False
    )
    scene = SimpleNamespace(entities={}, env_origins=torch.zeros((1, 3)))
    env = SimpleNamespace(
        num_envs=1, step_dt=dt, sim=sim, scene=scene, cfg=SimpleNamespace(viewer=cfg)
    )
    return env


def render_reference(
    model,
    qpos,
    dt,
    output,
    fps=30,
    width=800,
    height=600,
    preview_type="Kinematic reference preview (no physics rollout)",
):
    """Write MP4 and a six-sample contact sheet of a prescribed qpos trajectory.

    Parameters match raw MuJoCo: model is MjModel, qpos is [T, model.nq], dt is
    seconds between qpos samples. Output must end in .mp4. Returns paths/counts.
    Camera remains fixed and front-oblique. Geom group 0 (floor) and group 2
    (appearance) are visible; group 1 collision hulls and sites are hidden.
    Uses private renderer access only to capture stills from the SAME rendering
    implementation as ReplayVideo. All movie encoding remains in ReplayVideo.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    dt = float(dt)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) < 1:
        raise ValueError(f"qpos must have shape [T,{model.nq}], T >= 1; got {qpos.shape}")
    if not np.isfinite(qpos).all() or not math.isfinite(dt) or dt <= 0:
        raise ValueError("qpos must be finite and dt positive/finite")
    output = Path(output)
    env = _make_facade(model, qpos, dt, width, height)
    sample_ids = set(np.linspace(0, len(qpos) - 1, min(6, len(qpos)), dtype=int).tolist())
    stills = []
    qvel = np.zeros(model.nv)

    def assign(index):
        env.sim.data.qpos[0].copy_(torch.from_numpy(qpos[index]))
        if index + 1 < len(qpos):
            mujoco.mj_differentiatePos(model, qvel, dt, qpos[index], qpos[index + 1])
        else:
            qvel[:] = 0
        env.sim.data.qvel[0].copy_(torch.from_numpy(qvel))
        env.sim.data.time[0] = index * dt

    assign(0)
    with ReplayVideo(
        env,
        output,
        fps=fps,
        width=width,
        height=height,
        camera_distance=0.75,
        azimuth=145,
        elevation=-25,
        follow=True,
    ) as video:
        for index in range(len(qpos)):
            if index:
                assign(index)
                video.update()
            if index in sample_ids:
                video._renderer.update(env.sim.data)
                stills.append((index, np.array(video._renderer.render(), copy=True)))
        frame_count = video.frame_count

    # Still-image layout is not an alternate video-rendering or encoding path.
    thumb_w = min(400, width)
    thumb_h = round(height * thumb_w / width)
    cols = min(3, len(stills))
    rows = math.ceil(len(stills) / cols)
    header, caption = 34, 24
    sheet = Image.new("RGB", (cols * thumb_w, header + rows * (thumb_h + caption)), "#202733")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), preview_type, fill="white")
    for cell, (index, frame) in enumerate(stills):
        x, y = (cell % cols) * thumb_w, header + (cell // cols) * (thumb_h + caption)
        image = Image.fromarray(frame).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + thumb_h + 5), f"t={index * dt:.2f}s | sample {index}", fill="white")
    sheet_path = output.with_suffix(".contact_sheet.png")
    sheet.save(sheet_path)
    return {
        "video": str(output),
        "contact_sheet": str(sheet_path),
        "trajectory_samples": len(qpos),
        "video_frames": frame_count,
        "trajectory_duration_s": (len(qpos) - 1) * dt,
        "video_fps": fps,
        "preview_type": preview_type,
    }
