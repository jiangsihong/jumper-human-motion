"""View v2 target poses or recorded physics; no training or hidden root support."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = {"1": "01_side_step", "2": "02_kick_step", "3": "03_arm_body_sway"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", choices=list(NAMES), default="2")
    parser.add_argument("--mode", choices=["planned", "physics"], default="physics")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    suffix = "robot_trajectory" if args.mode == "planned" else "pd_rollout"
    with np.load(ROOT / "mapping" / f"{NAMES[args.motion]}_{suffix}.npz") as z:
        poses = z["qpos"]
        dt = float(z["dt"])
    model = mujoco.MjModel.from_xml_path(str(ROOT / "mapping/robot_preview.xml"))
    data = mujoco.MjData(model)
    print(
        f"V2 {args.mode.upper()} playback. No training. Physics mode replays the recorded unassisted PD execution."
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 115
        viewer.cam.elevation = -20
        viewer.cam.distance = 0.78
        viewer.cam.lookat[:] = [-0.025, 0, 0.12]
        viewer.opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
        start = time.monotonic()
        while viewer.is_running():
            index = int((time.monotonic() - start) / dt)
            index = index % len(poses) if args.loop else min(index, len(poses) - 1)
            with viewer.lock():
                data.qpos[:] = poses[index]
                data.qvel[:] = 0
                data.time = index * dt
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
