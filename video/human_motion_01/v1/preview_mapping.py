"""View prescribed robot references; this program never trains a policy."""

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
    parser.add_argument("--motion", choices=list(NAMES), default="1")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--mode", choices=["physics", "planned"], default="physics")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "mapping/robot_preview.xml"))
    data = mujoco.MjData(model)
    suffix = "pd_rollout" if args.mode == "physics" else "robot_trajectory"
    with np.load(ROOT / "mapping" / f"{NAMES[args.motion]}_{suffix}.npz") as z:
        qpos = z["qpos"]
        dt = float(z["dt"])
    if args.mode == "physics":
        print("RECORDED PHYSICS PLAYBACK: replaying a completed free-base PD probe; no learning.")
    else:
        print("KINEMATIC TARGET PREVIEW: root and joints are prescribed; no physics or learning.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.75
        viewer.cam.lookat[:] = [0, 0, 0.12]
        viewer.opt.geomgroup[:] = [1, 0, 1, 0, 0, 0]
        start = time.monotonic()
        while viewer.is_running():
            elapsed = time.monotonic() - start
            index = int(elapsed / dt)
            index = index % len(qpos) if args.loop else min(index, len(qpos) - 1)
            with viewer.lock():
                data.qpos[:] = qpos[index]
                data.qvel[:] = 0
                data.time = index * dt
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
