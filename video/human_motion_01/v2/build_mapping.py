"""Expressive video-to-Jumper references. No policy training is performed.

Human arms drive only LF/RF; human foot observations drive only LM/RM/LR/RR.
Four-leg steps use an explicit crawl contact schedule. The observed right-foot
kick is adapted to a large RR backswing/forward-swing trajectory. Candidate
references are never replaced by a smaller-amplitude PD playback.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from tasks.jumper.common.constants import LEGS
from tasks.jumper.common.dance.motion import LEG_JOINTS, load_source, ensure_motion_npz

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "v1"
DT = 0.02
TIME_SCALE = 2.0
TIME_SCALES = {"01_side_step": 2.0, "02_kick_step": 2.0, "03_arm_body_sway": 2.5}
JOBS = [("01_side_step", 195, 2.8), ("02_kick_step", 123, 12.4), ("03_arm_body_sway", 103, 24.9)]
SETTINGS = dict(
    version=2,
    time_scale=TIME_SCALE,
    control_hz=50,
    arms_only=["LF", "RF"],
    feet_only=["LM", "RM", "LR", "RR"],
    step_clearance_m=0.045,
    sway_step_clearance_m=0.055,
    right_rear_kick_x_range_m=[-0.20, -0.04],
    right_rear_kick_height_m=0.060,
    right_rear_kick_outward_m=0.020,
    max_joint_speed_rad_s=12.0,
    time_scales=TIME_SCALES,
    source_pose_note="estimated monocular landmarks; kick depth adapted from reviewed phases",
    priority="preserve expressive step/kick targets; report execution failures separately",
)

# The v1 numerical IK utility is shared by filename, without modifying project
# package resolution or importing the artifact folder as a training task.
spec = importlib.util.spec_from_file_location(
    "reviewed_mapping_ik", SOURCE / "build_motion_mapping.py"
)
utility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility)
utility.PARAMS["front_elbow_clearance_x_m"] = 0.085


def smooth(a):
    return savgol_filter(a, 7, 2, axis=0, mode="interp")


def ease(u):
    return u * u * (3 - 2 * u)


def observations(name, count):
    with np.load(SOURCE / "poses" / name / "human_landmarks.npz") as z:
        p = z["image_landmarks"][:count].astype(float)
        w = z["world_landmarks"][:count].astype(float)
        valid = (z["visibility"][:count] >= 0.5) & (z["presence"][:count] >= 0.5)
    valid &= np.isfinite(p).all(axis=2) & np.isfinite(w).all(axis=2)
    frame = np.arange(count)
    for j in range(33):
        if not valid[:, j].any():
            raise ValueError(f"{name}: no usable landmark {j}")
        for axis in range(3):
            p[:, j, axis] = np.interp(frame, frame[valid[:, j]], p[valid[:, j], j, axis])
            w[:, j, axis] = np.interp(frame, frame[valid[:, j]], w[valid[:, j], j, axis])
    p = smooth(p)
    w = smooth(w)
    xy = p[:, :, :2] * [1026.0, 720.0]
    shoulder = (xy[:, 11] + xy[:, 12]) / 2
    hip = (xy[:, 23] + xy[:, 24]) / 2
    torso = max(40.0, float(np.median(np.linalg.norm(shoulder - hip, axis=1))))
    center0 = np.median(hip[:4], axis=0)
    r = np.arctan2(shoulder[:, 0] - hip[:, 0], hip[:, 1] - shoulder[:, 1])
    ws = (w[:, 11] + w[:, 12]) / 2
    wh = (w[:, 23] + w[:, 24]) / 2
    pitch = -(ws[:, 2] - wh[:, 2]) / np.maximum(np.linalg.norm(ws - wh, axis=1), 0.1)
    yaw = np.arctan2(w[:, 11, 2] - w[:, 12, 2], np.abs(w[:, 11, 0] - w[:, 12, 0]) + 0.05)
    f = dict(
        body_y=np.clip((hip[:, 0] - center0[0]) / torso * 0.20, -0.065, 0.065),
        body_z=np.clip((center0[1] - hip[:, 1]) / torso * 0.08, 0, 0.020),
        roll=np.clip(-0.55 * (r - np.median(r[:4])), -0.18, 0.18),
        pitch=np.clip(0.22 * (pitch - np.median(pitch[:4])), -0.14, 0.14),
        yaw=np.clip(0.15 * (yaw - np.median(yaw[:4])), -0.14, 0.14),
    )
    for side, s, e, wr, h, k, a in [
        ("left", 11, 13, 15, 23, 25, 27),
        ("right", 12, 14, 16, 24, 26, 28),
    ]:
        length = np.maximum(
            np.linalg.norm(xy[:, s] - xy[:, e], axis=1)
            + np.linalg.norm(xy[:, e] - xy[:, wr], axis=1),
            torso * 0.7,
        )
        sign = 1 if side == "left" else -1
        f[side + "_hand_up"] = (xy[:, s, 1] - xy[:, wr, 1]) / length
        f[side + "_hand_out"] = sign * (xy[:, wr, 0] - xy[:, s, 0]) / length
        u = xy[:, s] - xy[:, e]
        v = xy[:, wr] - xy[:, e]
        f[side + "_elbow_bend"] = np.pi - np.arccos(
            np.clip(
                (u * v).sum(axis=1)
                / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-6),
                -1,
                1,
            )
        )
        rel = (xy[:, a, 0] - hip[:, 0]) / torso
        f[side + "_stance_out"] = np.clip((rel - np.median(rel[:4])) * 0.12, -0.020, 0.020)
        depth = w[:, a, 2] - wh[:, 2]
        f[side + "_foot_forward"] = np.clip(-(depth - np.median(depth[:4])) * 0.16, -0.025, 0.025)
        thigh = xy[:, h] - xy[:, k]
        shin = xy[:, a] - xy[:, k]
        f[side + "_knee_bend"] = np.pi - np.arccos(
            np.clip(
                (thigh * shin).sum(axis=1)
                / (np.linalg.norm(thigh, axis=1) * np.linalg.norm(shin, axis=1) + 1e-6),
                -1,
                1,
            )
        )
    return f


def plan_feet(name, time_source, feature):
    # A compact four-foot stance extends the support region under the body.
    neutral = np.array(
        [
            [0.16, 0.08, 0.011],
            [0.16, -0.08, 0.011],
            [0.000, 0.210, 0.010],
            [0.000, -0.210, 0.010],
            [-0.105, 0.210, 0.010],
            [-0.105, -0.210, 0.010],
        ]
    )
    feet = np.repeat(neutral[None], len(time_source), axis=0)
    swing = np.zeros((len(time_source), 6), bool)
    events = []
    if name == "01_side_step":
        bouts = [(0.22, 1.50), (1.9, 3.05), (3.45, 4.80), (5.10, time_source[-1])]
        clearance = 0.045
    elif name == "03_arm_body_sway":
        bouts = [(0.18, 1.12), (1.15, 2.12), (2.15, 3.02)]
        clearance = 0.055
    else:
        bouts = []
        clearance = 0.04
    for begin, end in bouts:
        direction = np.interp(end, time_source, feature["body_y"]) - np.interp(
            begin, time_source, feature["body_y"]
        )
        order = [2, 5, 3, 4] if direction >= 0 else [3, 4, 2, 5]
        for slot, leg in enumerate(order):
            a = begin + (end - begin) * slot / 4
            b = begin + (end - begin) * (slot + 1) / 4
            events.append((a, b, leg, clearance))
    if name == "02_kick_step":
        # RM performs the observed preparation; RR carries the large kick.
        events = [(0.40, 1.02, 3, 0.040)]
    for begin, end, leg, clearance in sorted(events):
        idx = np.flatnonzero((time_source >= begin) & (time_source <= end))
        if not len(idx):
            continue
        side = "left" if LEGS[leg].startswith("L") else "right"
        start = feet[max(0, idx[0] - 1), leg].copy()
        goal = neutral[leg].copy()
        if name == "03_arm_body_sway":
            goal[0] += 0.5 * np.interp(end, time_source, feature[side + "_foot_forward"])
        if name != "02_kick_step":
            goal[1] += np.interp(end, time_source, feature["body_y"]) + np.interp(
                end, time_source, feature[side + "_stance_out"]
            )
        # A recovery/preparation step still has a visible lift even when its
        # final support point is near the previous support point.
        u = (time_source[idx] - begin) / (end - begin)
        path = start[None] + ease(u)[:, None] * (goal - start)[None]
        path[:, 2] += clearance * np.sin(np.pi * u) ** 2
        feet[idx, leg] = path
        swing[idx, leg] = True
        feet[idx[-1] + 1 :, leg] = goal
    if name == "02_kick_step":
        # Reviewed right-foot phase: outward/backward retraction, then forward
        # swing and return. RR's outward workspace permits a 160 mm X stroke.
        knots = np.array([0.0, 1.08, 1.90, 2.18, 2.92, 3.43, time_source[-1]])
        positions = np.array(
            [
                neutral[5],
                neutral[5],
                [-0.20, -0.23, 0.070],
                [-0.20, -0.23, 0.070],
                [-0.04, -0.23, 0.070],
                neutral[5],
                neutral[5],
            ]
        )
        for k in range(len(knots) - 1):
            idx = np.flatnonzero((time_source >= knots[k]) & (time_source <= knots[k + 1]))
            if not len(idx):
                continue
            u = (time_source[idx] - knots[k]) / (knots[k + 1] - knots[k])
            feet[idx, 5] = positions[k] + ease(u)[:, None] * (positions[k + 1] - positions[k])
        swing[:, 5] = (time_source > 1.08) & (time_source < 3.43)
    return neutral, feet, swing, events


def make_model():
    tree = ET.parse(SOURCE / "mapping/robot_preview.xml")
    tree.getroot().set("model", "jumper_expressive_mapping_v2")
    path = ROOT / "mapping/robot_preview.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(path))


def audit(model, qpos, robot):
    data = mujoco.MjData(model)
    floor = model.geom("mapping_floor").id
    feet = []
    floor_max = self_max = 0.0
    worst = []
    for k, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        feet.append(data.site_xpos[robot.sites].copy())
        for c in data.contact:
            if c.dist < 0:
                if floor in (c.geom1, c.geom2):
                    floor_max = max(floor_max, -float(c.dist))
                else:
                    self_max = max(self_max, -float(c.dist))
                    if c.dist < -0.003:
                        worst.append(
                            [
                                k * DT,
                                model.geom(c.geom1).name,
                                model.geom(c.geom2).name,
                                float(c.dist),
                            ]
                        )
    return np.asarray(feet), dict(
        max_floor_overlap_m=floor_max,
        max_self_overlap_m=self_max,
        contacts_over_3mm=worst[:15],
        note="prescribed geometry audit; not a dynamic stability certificate",
    )


def build_one(robot, name, count, offset, render):
    raw = observations(name, count)
    time_scale = TIME_SCALES[name]
    source_time = np.arange(0, (count - 1) / 30 * time_scale + DT / 2, DT) / time_scale
    source_time = np.minimum(source_time, (count - 1) / 30)
    feature = {k: PchipInterpolator(np.arange(count) / 30, v)(source_time) for k, v in raw.items()}
    neutral, feet, swing, events = plan_feet(name, source_time, feature)
    q = robot.home.copy()
    poses = []
    errors = []
    root_adjust = []
    targets = []
    for k, t in enumerate(source_time):
        robot.step_low = (
            robot.low.copy()
            if k == 0
            else np.maximum(robot.low, q - SETTINGS["max_joint_speed_rad_s"] * DT)
        )
        robot.step_high = (
            robot.high.copy()
            if k == 0
            else np.minimum(robot.high, q + SETTINGS["max_joint_speed_rad_s"] * DT)
        )
        target = feet[k].copy()
        # Advance the base as the planted footsteps advance. Translating every
        # foot along with an unactuated root would create an in-place PD replay.
        stance_center = float(np.mean(feet[k, 2:, 1] - neutral[2:, 1]))
        xyz = np.array(
            [-0.055, stance_center + 0.10 * feature["body_y"][k], 0.105 + feature["body_z"][k]]
        )
        rpy = np.array([feature["roll"][k], feature["pitch"][k], feature["yaw"][k]])
        R = Rotation.from_euler("xyz", rpy).as_matrix()
        # High lifts need an outward path to stay inside the actual three-joint
        # leg workspace. This changes swing direction, not requested clearance.
        for j in [2, 3, 4, 5]:
            if swing[k, j] and not (name == "02_kick_step" and j == 5):
                clearance = (
                    (0.040 if j == 3 else 0.035)
                    if name == "02_kick_step"
                    else (0.045 if name == "01_side_step" else 0.055)
                )
                alpha = np.clip((target[j, 2] - 0.010) / clearance, 0, 1)
                sign = 1 if LEGS[j].startswith("L") else -1
                rel = np.array([0.035 if j < 4 else -0.050, sign * 0.230, 0.0])
                reachable = xyz[:2] + (R @ rel)[:2]
                target[j, :2] = (1 - alpha) * target[j, :2] + alpha * reachable
        for j, side in enumerate(["left", "right"]):
            sign = 1 if j == 0 else -1
            up = float(np.clip(feature[side + "_hand_up"][k], -1, 1))
            outward = float(np.clip(feature[side + "_hand_out"][k], -0.25, 1))
            elbow = feature[side + "_elbow_bend"][k] / np.pi
            rel = np.array(
                [
                    0.230 + 0.015 * max(outward, 0) - 0.045 * max(up - 0.35, 0) - 0.010 * elbow,
                    sign * np.clip(0.080 + 0.055 * outward, 0.060, 0.135),
                    np.clip(-0.090 + 0.130 * (up + 0.90), -0.090, 0.130),
                ]
            )
            target[j] = xyz + R @ rel
        q = robot.solve(xyz, rpy, target, q)
        # Shift the root over the THREE planted leg targets, without reducing
        # the active foot lift or X stroke. Arm targets follow the shared root.
        support = np.array([j for j in [2, 3, 4, 5] if not swing[k, j]])
        shift = np.zeros(2)
        if len(support) >= 3:
            for _ in range(2):
                com = robot.d.subtree_com[robot.m.body("base_link").id, :2]
                projected = utility.inside_polygon(com, target[support, :2], 0.005)
                delta = np.clip(projected - com, [-0.008, -0.010], [0.008, 0.010])
                if np.linalg.norm(delta) < 0.0005:
                    break
                xyz[:2] += delta
                shift += delta
                target[:2, :2] += delta
                q = robot.solve(xyz, rpy, target, q)
        # Adapt the horizontal foot placement locally when the original point
        # is outside the real joint workspace. Keep every requested lift height.
        # This is not the old global amplitude clamp: an elevated leg moves
        # farther outward; a stance point moves to the nearest attainable XY.
        for correction in range(10):
            actual = robot.d.site_xpos[robot.sites].copy()
            changed = False
            for j in [2, 3, 4, 5]:
                error = np.linalg.norm(actual[j] - target[j])
                if error < 0.0015:
                    continue
                if swing[k, j]:
                    sign = 1 if LEGS[j].startswith("L") else -1
                    target[j, 1] += sign * 0.006
                    if j < 4:
                        target[j, 0] = 0.65 * target[j, 0] + 0.35 * (xyz[0] + 0.010)
                else:
                    target[j, :2] = 0.3 * target[j, :2] + 0.7 * actual[j, :2]
                changed = True
            if not changed:
                break
            q = robot.solve(xyz, rpy, target, q)
        mujoco.mj_forward(robot.m, robot.d)
        # Correct mesh-pad penetration, never shrink a requested swing.
        floor = robot.m.geom("mapping_floor").id
        total_rise = np.zeros(6)
        for _ in range(6):
            rise = np.zeros(6)
            for c in robot.d.contact:
                if floor not in (c.geom1, c.geom2) or c.dist >= -0.0005:
                    continue
                gid = c.geom2 if c.geom1 == floor else c.geom1
                gname = robot.m.geom(gid).name
                for j, leg in enumerate(LEGS):
                    if gname.startswith(leg + "_"):
                        rise[j] = max(rise[j], -float(c.dist) + 0.0002)
            rise = np.minimum(rise, np.maximum(0.010 - total_rise, 0.0))
            if rise.max() < 0.0005:
                break
            total_rise += rise
            target[:, 2] += rise
            q = robot.solve(xyz, rpy, target, q)
            mujoco.mj_forward(robot.m, robot.d)
        errors.append(np.linalg.norm(robot.d.site_xpos[robot.sites] - target, axis=1))
        targets.append(target.copy())
        poses.append(robot.d.qpos.copy())
        root_adjust.append(shift)
    poses = np.asarray(poses)
    # Hold the first and last poses briefly: no invisible amplitude-reducing
    # return to v1's neutral trajectory is appended.
    lead = np.repeat(poses[:1], round(0.5 / DT), axis=0)
    tail = np.repeat(poses[-1:], round(0.5 / DT), axis=0)
    full = np.concatenate([lead, poses, tail])
    time = np.arange(len(full)) * DT
    velocities = np.zeros((len(full), robot.m.nv))
    for k in range(len(full) - 1):
        mujoco.mj_differentiatePos(robot.m, velocities[k], DT, full[k], full[k + 1])
    actual_feet, geometry = audit(robot.m, full, robot)
    qj = full[:, robot.adr]
    detailed = ROOT / "mapping" / f"{name}_robot_trajectory.npz"
    np.savez_compressed(
        detailed,
        qpos=full.astype("f4"),
        qvel=velocities.astype("f4"),
        dt=np.float64(DT),
        time_scale=np.float64(time_scale),
        joint_names=np.array(robot.names),
        feet_actual=actual_feet.astype("f4"),
        foot_targets=np.asarray(targets, dtype="f4"),
        motion_start_index=len(lead),
        motion_end_index=len(lead) + len(poses),
        source_time=(offset + source_time).astype("f4"),
    )
    directory = ROOT / "trajectories" / name
    directory.mkdir(parents=True, exist_ok=True)
    rpy = Rotation.from_quat(full[:, [4, 5, 6, 3]]).as_euler("xyz")
    channels = dict(
        time=time.astype("f4"),
        dt=np.float64(DT),
        provenance_json=np.array(
            json.dumps(
                dict(
                    settings=SETTINGS,
                    source_name=name,
                    reference_kind="expressive kinematic candidate, not a learned or measured trajectory",
                    measured_alias_note="meas_* are source-schema aliases of the planned kinematic pose",
                )
            )
        ),
    )
    for prefix in ["", "meas_"]:
        for j, a in enumerate(["x", "y", "z"]):
            channels[prefix + "body_" + a] = full[:, j].astype("f4")
        for j, a in enumerate(["roll", "pitch", "yaw"]):
            channels[prefix + "body_" + a] = rpy[:, j].astype("f4")
        for leg, names in LEG_JOINTS.items():
            for j, n in enumerate(names):
                channels[f"{prefix}L{leg}_j{j}"] = qj[:, robot.names.index(n)].astype("f4")
    path = directory / "reference.npz"
    np.savez_compressed(path, **channels)
    loaded = load_source(path)
    assert len(loaded.joint_pos) == len(full)
    try:
        converted = ensure_motion_npz(DT, directory, reference="measured")
        conversion = dict(status="passed", path=str(converted))
    except ValueError as e:
        conversion = dict(
            status="not_compatible_with_fixed_four_support_dance_check",
            reason=str(e),
            note="Do not lower kick/step amplitude to satisfy an all-four-feet-grounded dance assumption; future task needs scheduled contacts.",
        )
    with (ROOT / "mapping" / f"{name}_joints.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["time_s", *robot.names])
        w.writerows(np.column_stack([time, qj]))
    ranges = np.ptp(actual_feet, axis=0)
    report = dict(
        name=name,
        version=2,
        training_started=False,
        samples=len(full),
        duration_s=time[-1],
        primary_reference_type="expressive kinematic candidate, not overwritten by PD playback",
        leg_mapping="hands LF/RF only; feet LM/RM/LR/RR only",
        foot_world_excursion_m={
            leg: dict(x=float(ranges[j, 0]), y=float(ranges[j, 1]), z=float(ranges[j, 2]))
            for j, leg in enumerate(LEGS)
        },
        body_world_excursion_m=np.ptp(full[:, :3], axis=0).tolist(),
        peak_joint_speed_rad_s=float(np.abs(velocities[:, robot.dof]).max()),
        min_joint_limit_margin_rad=float(
            np.min(
                np.minimum(
                    qj - robot.m.jnt_range[robot.jids, 0], robot.m.jnt_range[robot.jids, 1] - qj
                )
            )
        ),
        ik_max_error_m=float(np.max(errors)),
        ik_mean_error_m=float(np.mean(errors)),
        geometry=geometry,
        format_load_check="passed",
        fixed_support_dance_conversion=conversion,
        source_interval_s=[offset, offset + count / 30],
        step_events=events,
        kick_note="right foot RR: retract outward/back, swing forward, return"
        if name.startswith("02")
        else None,
    )
    if render:
        from jumper_mapping_renderer import render_reference

        report["preview"] = render_reference(
            robot.m,
            full,
            DT,
            ROOT / "previews" / f"{name}_mapped.mp4",
            preview_type="V2 expressive mapping target - no learning - no physics integration",
        )
    (ROOT / "mapping" / f"{name}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    robot = utility.Robot(make_model())
    rules = dict(
        schema="jumper-human-retarget/2",
        settings=SETTINGS,
        mapping=[
            dict(human="left arm", robot=["LF"]),
            dict(human="right arm", robot=["RF"]),
            dict(human="left foot", robot=["LM", "LR"]),
            dict(human="right foot", robot=["RM", "RR"]),
        ],
        adaptations=[
            "one swinging leg at a time during crawl steps",
            "RR is the principal right-leg kick because its joint workspace permits a much larger swing",
            "body compensation moves the root, without suppressing the leg target",
            "monocular depth is adapted from reviewed kick phases, not claimed as ground truth",
        ],
        previous_version="../v1/mapping and ../v1/previews are preserved",
    )
    (ROOT / "mapping/mapping_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    reports = [build_one(robot, *job, args.render) for job in JOBS]
    (ROOT / "mapping/validation_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
