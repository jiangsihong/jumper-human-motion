"""Bounded free-base PD tracking probe, with gravity and without learning.

This is a diagnostic of the preview's simple 10/.5 PD actuators. It is not the
project's torque-speed actuator model, a learned policy, or hardware validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tasks.jumper.common.dance.motion import LEG_JOINTS, ensure_motion_npz

ROOT = Path(__file__).resolve().parent


def main():
    reports = []
    for name in ["01_side_step", "02_kick_step", "03_arm_body_sway"]:
        model = mujoco.MjModel.from_xml_path(str(ROOT / "mapping/robot_preview.xml"))
        data = mujoco.MjData(model)
        with np.load(ROOT / "mapping" / f"{name}_robot_trajectory.npz") as z:
            target = z["qpos"].astype(float)
            dt = float(z["dt"])
            joint_names = z["joint_names"].tolist()
        adr = np.array([model.joint(n).qposadr[0] for n in joint_names])
        base = model.body("base_link").id
        data.qpos[:] = target[0]
        data.ctrl[:] = target[0, adr]
        mujoco.mj_forward(model, data)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        actual = []
        error = []
        up = []
        height = []
        first_fall = None
        saturation = []
        for k, q in enumerate(target):
            data.ctrl[:] = q[adr]
            for _ in range(round(dt / model.opt.timestep)):
                mujoco.mj_step(model, data)
            actual.append(data.qpos.copy())
            error.append(data.qpos[adr] - q[adr])
            up.append(data.xmat[base].reshape(3, 3)[2, 2])
            height.append(data.qpos[2])
            saturation.append(np.mean(np.abs(data.actuator_force) > 0.99 * 1.7464))
            if first_fall is None and (height[-1] < 0.055 or up[-1] < 0.65):
                first_fall = k * dt
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite physics state")
        actual = np.asarray(actual)
        error = np.asarray(error)
        np.savez_compressed(
            ROOT / "mapping" / f"{name}_pd_rollout.npz",
            qpos=actual.astype("f4"),
            dt=np.float64(dt),
            joint_error=error.astype("f4"),
            root_up_cos=np.array(up, dtype="f4"),
        )
        root_xy_rmse = float(np.sqrt(np.mean((actual[:, :2] - target[:, :2]) ** 2)))
        # The primary training-format reference uses states actually reached by
        # free-base physics, not the requested root pose. Preserve the original
        # planned channels and its provenance alongside those observations.
        reference = ROOT / "trajectories" / name / "reference.npz"
        with np.load(reference) as z:
            channels = {k: z[k].copy() for k in z.files}
        draft = ROOT / "mapping" / f"{name}_kinematic_source.npz"
        np.savez_compressed(draft, **channels)
        rpy = Rotation.from_quat(actual[:, [4, 5, 6, 3]]).as_euler("xyz")
        for j, axis in enumerate(["x", "y", "z"]):
            channels["meas_body_" + axis] = actual[:, j].astype("f4")
        for j, axis in enumerate(["roll", "pitch", "yaw"]):
            channels["meas_body_" + axis] = rpy[:, j].astype("f4")
        for leg, names in LEG_JOINTS.items():
            for j, jname in enumerate(names):
                channels[f"meas_L{leg}_j{j}"] = actual[:, model.joint(jname).qposadr[0]].astype(
                    "f4"
                )
        provenance = json.loads(str(channels["provenance_json"]))
        provenance["reference_kind"] = "MuJoCo simple-PD physically measured reference"
        provenance["measured_channel_note"] = (
            "meas_* contain free-base MuJoCo states reached with gravity and resolved contacts; no external root forces"
        )
        provenance["simulation"] = (
            "preview XML simple kp10 kv.5 torque-limited position actuators; not full project servo curve"
        )
        channels["provenance_json"] = np.array(json.dumps(provenance))
        np.savez_compressed(reference, **channels)
        converted = ensure_motion_npz(0.02, reference.parent, reference="measured")
        report = dict(
            name=name,
            probe="free-base simple PD kp=10 kv=.5, +/-1.7464Nm; gravity on; no root forces",
            steps=len(target),
            settling_s=1.0,
            duration_s=len(target) * dt,
            first_fall_time_s=first_fall,
            min_base_height_m=float(min(height)),
            min_up_cos=float(min(up)),
            joint_rmse_rad=float(np.sqrt(np.mean(error**2))),
            root_xy_rmse_m=root_xy_rmse,
            mean_actuator_saturation_fraction=float(np.mean(saturation)),
            status="fell during untrained PD playback"
            if first_fall is not None
            else "no fall in this bounded simple-PD probe",
            measured_reference=str(reference),
            converted_reference=str(converted),
            measured_reference_format_check="passed",
            note="Does not establish deployability or policy success; PD target errors and base drift remain.",
        )
        reports.append(report)
        print(json.dumps(report), flush=True)
    (ROOT / "mapping" / "physics_probe.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
