"""Measure free-base PD execution of v2, without changing its expressive target."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ["01_side_step", "02_kick_step", "03_arm_body_sway"]
LEGS = ["LF", "RF", "LM", "RM", "LR", "RR"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=NAMES)
    args = parser.parse_args()
    reports = []
    for name in NAMES:
        if args.only and name != args.only:
            continue
        model = mujoco.MjModel.from_xml_path(str(ROOT / "mapping/robot_preview.xml"))
        kp, kv = 10.0, 0.5
        model.actuator_gainprm[:, 0] = kp
        model.actuator_biasprm[:, 1] = -kp
        model.actuator_biasprm[:, 2] = -kv
        data = mujoco.MjData(model)
        with np.load(ROOT / "mapping" / f"{name}_robot_trajectory.npz") as z:
            target = z["qpos"].astype(float)
            dt = float(z["dt"])
            names = z["joint_names"].tolist()
        adr = np.array([model.joint(n).qposadr[0] for n in names])
        sites = [model.site(n).id for n in LEGS]
        base = model.body("base_link").id
        data.qpos[:] = target[0]
        data.ctrl[:] = target[0, adr]
        mujoco.mj_forward(model, data)
        for _ in range(1000):
            mujoco.mj_step(model, data)
        actual = []
        feet = []
        local = []
        error = []
        up = []
        height = []
        first_fall = None
        for k, q in enumerate(target):
            data.ctrl[:] = q[adr]
            for _ in range(round(dt / model.opt.timestep)):
                mujoco.mj_step(model, data)
            actual.append(data.qpos.copy())
            feet.append(data.site_xpos[sites].copy())
            R = data.xmat[base].reshape(3, 3)
            local.append((data.site_xpos[sites] - data.xpos[base]) @ R)
            error.append(data.qpos[adr] - q[adr])
            up.append(R[2, 2])
            height.append(data.qpos[2])
            if first_fall is None and (height[-1] < 0.055 or up[-1] < 0.65):
                first_fall = k * dt
            if not np.isfinite(data.qpos).all():
                raise RuntimeError("Non-finite physics state")
        actual = np.asarray(actual)
        feet = np.asarray(feet)
        local = np.asarray(local)
        error = np.asarray(error)
        np.savez_compressed(
            ROOT / "mapping" / f"{name}_pd_rollout.npz",
            qpos=actual.astype("f4"),
            feet_actual=feet.astype("f4"),
            feet_in_body=local.astype("f4"),
            dt=np.float64(dt),
            joint_error=error.astype("f4"),
            root_up_cos=np.array(up, dtype="f4"),
        )
        span = np.ptp(feet, axis=0)
        local_span = np.ptp(local, axis=0)
        report = dict(
            name=name,
            version=2,
            training_started=False,
            mode=f"free-base PD, gravity and contacts, no root assistance, kp{kp:g} kv{kv:g} torque1.7464Nm",
            kp=kp,
            kv=kv,
            first_fall_time_s=first_fall,
            min_base_height_m=float(min(height)),
            min_up_cos=float(min(up)),
            joint_rmse_rad=float(np.sqrt(np.mean(error**2))),
            root_xy_span_m=np.ptp(actual[:, :2], axis=0).tolist(),
            feet_world_span_m={leg: span[j].tolist() for j, leg in enumerate(LEGS)},
            feet_relative_body_span_m={leg: local_span[j].tolist() for j, leg in enumerate(LEGS)},
            status="fall observed; this is not a trained controller"
            if first_fall is not None
            else "no fall in this bounded PD execution",
            target_policy="Expressive target remains unchanged. No reduced-amplitude PD result is substituted for it.",
        )
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    report_path = ROOT / "mapping/physics_report.json"
    if args.only and report_path.exists():
        old = {r["name"]: r for r in json.loads(report_path.read_text())}
        old.update({r["name"]: r for r in reports})
        reports = [old[n] for n in NAMES]
    report_path.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
