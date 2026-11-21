"""Video-observation-driven, conservative kinematic retargeting for Jumper.

Produces candidate references, not a learned policy. Visibility masks, retiming,
support adaptations and measured kinematic checks are recorded explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from tasks.jumper.common.constants import HOME, STAND_Z, LEGS
from tasks.jumper.common.dance.motion import LEG_JOINTS, ensure_motion_npz

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
DT = 0.02
PARAMS = dict(
    control_hz=50,
    source_fps=30,
    playback_time_scale=1.8,
    lead_in_s=1.0,
    lead_out_s=1.0,
    final_hold_s=0.5,
    visibility_threshold=0.5,
    smoothing_window_frames=7,
    max_body_roll_rad=0.12,
    max_body_pitch_rad=0.10,
    max_body_yaw_rad=0.10,
    max_body_lateral_m=0.022,
    support_backshift_m=0.060,
    max_front_lift_m=0.15,
    max_front_outward_m=0.045,
    max_leg_lift_m=0.0035,
    max_leg_lateral_m=0.009,
    support_margin_m=0.009,
    joint_limit_margin_rad=0.025,
    ik_target_tolerance_m=0.0015,
    max_joint_speed_rad_s=4.5,
    front_elbow_clearance_x_m=0.065,
)
JOBS = [("01_side_step", 195, 2.8), ("02_kick_step", 123, 12.4), ("03_arm_body_sway", 103, 24.9)]


def smooth(a):
    return savgol_filter(a, PARAMS["smoothing_window_frames"], 2, axis=0, mode="interp")


def rotation(rpy):
    return Rotation.from_euler("xyz", rpy)


def features(name, count):
    with np.load(ROOT / "poses" / name / "human_landmarks.npz") as z:
        image = z["image_landmarks"][:count].astype(float)
        world = z["world_landmarks"][:count].astype(float)
        visibility = z["visibility"][:count]
        presence = z["presence"][:count]
    valid = (
        (visibility >= 0.5)
        & (presence >= 0.5)
        & np.isfinite(image).all(axis=2)
        & np.isfinite(world).all(axis=2)
    )
    frames = np.arange(count)
    # Low-visibility observations are not used as direct targets. Interior gaps
    # interpolate; leading/trailing gaps hold the nearest valid observation.
    for j in range(33):
        good = valid[:, j]
        if not good.any():
            raise ValueError(f"{name}: landmark {j} is never visible")
        for axis in range(3):
            image[:, j, axis] = np.interp(frames, frames[good], image[good, j, axis])
            world[:, j, axis] = np.interp(frames, frames[good], world[good, j, axis])
    image = smooth(image)
    world = smooth(world)
    xy = image[:, :, :2] * np.array([1026.0, 720.0])
    shoulder = (xy[:, 11] + xy[:, 12]) / 2
    hip = (xy[:, 23] + xy[:, 24]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    scale = max(float(np.median(torso)), 30.0)
    roll = -np.arctan2(shoulder[:, 0] - hip[:, 0], hip[:, 1] - shoulder[:, 1])
    wshoulder = (world[:, 11] + world[:, 12]) / 2
    whip = (world[:, 23] + world[:, 24]) / 2
    wtorso = np.linalg.norm(wshoulder - whip, axis=1)
    pitch = -(wshoulder[:, 2] - whip[:, 2]) / np.maximum(wtorso, 0.1)
    yaw = np.arctan2(
        world[:, 11, 2] - world[:, 12, 2], np.abs(world[:, 11, 0] - world[:, 12, 0]) + 0.05
    )
    out = dict(
        roll=np.clip(0.35 * (roll - np.median(roll[:5])), -0.12, 0.12),
        pitch=np.clip(0.15 * (pitch - np.median(pitch[:5])), -0.10, 0.10),
        yaw=np.clip(0.12 * (yaw - np.median(yaw[:5])), -0.10, 0.10),
        body_y=np.clip(0.055 * (hip[:, 0] - np.median(hip[:5, 0])) / scale, -0.022, 0.022),
    )
    for side, s, e, w, h, k, a in [
        ("left", 11, 13, 15, 23, 25, 27),
        ("right", 12, 14, 16, 24, 26, 28),
    ]:
        # Anatomical sides come from landmark labels, not screen halves.
        lift = (hip[:, 1] - xy[:, w, 1]) / scale
        baseline = float(np.median(lift[:4]))
        out[f"{side}_hand_lift"] = np.clip((lift - baseline) * 0.105, 0, 0.15)
        outward = (xy[:, w, 0] - xy[:, s, 0]) / scale * (1 if side == "left" else -1)
        out[f"{side}_hand_out"] = np.clip(outward * 0.035, -0.015, 0.045)
        u = xy[:, s] - xy[:, e]
        v = xy[:, w] - xy[:, e]
        angle = np.arccos(
            np.clip(
                np.sum(u * v, axis=1)
                / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-6),
                -1,
                1,
            )
        )
        out[f"{side}_elbow_bend"] = np.pi - angle
        foot = (xy[:, a, 0] - hip[:, 0]) / scale
        out[f"{side}_foot_y"] = np.clip((foot - np.median(foot[:4])) * 0.020, -0.009, 0.009)
        up = (np.median(xy[:4, a, 1]) - xy[:, a, 1]) / scale
        thigh = xy[:, h] - xy[:, k]
        shin = xy[:, a] - xy[:, k]
        knee_angle = np.arccos(
            np.clip(
                np.sum(thigh * shin, axis=1)
                / (np.linalg.norm(thigh, axis=1) * np.linalg.norm(shin, axis=1) + 1e-6),
                -1,
                1,
            )
        )
        knee_bend = np.pi - knee_angle
        out[f"{side}_knee_bend"] = knee_bend
        knee_lift = np.maximum(knee_bend - np.median(knee_bend[:4]), 0) * 0.004
        out[f"{side}_foot_lift"] = np.clip(
            np.maximum(up * 0.030, knee_lift), 0, PARAMS["max_leg_lift_m"]
        )
    out["confidence"] = np.min(
        visibility[:, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]], axis=1
    )
    meta = dict(
        frames=count,
        source_start_s=dict((n, o) for n, c, o in JOBS)[name],
        invalid_landmark_samples=int(
            (~valid[:, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]).sum()
        ),
        observed_12_landmark_samples=count * 12,
        source_type="monocular estimated landmarks; filtered/interpolated where masked",
        third_clip_policy="first 103 frames only, before close-view cut"
        if name.startswith("03")
        else "all extracted frames",
    )
    with (ROOT / "poses" / name / "mapping_features.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", *out])
        for i in range(count):
            writer.writerow([i / 30, *[out[k][i] for k in out]])
    return out, meta


def make_model():
    source = REPO / "assets/jumper/jumper.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "jumper_video_motion_preview")
    output = ROOT / "mapping" / "robot_preview.xml"
    root.find("compiler").set("meshdir", os.path.relpath(source.parent, output.parent))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", ".001")
    option.set("integrator", "implicitfast")
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="mapping_floor",
        type="plane",
        size="2 2 .1",
        rgba=".26 .30 .34 1",
        friction="1 .01 .01",
    )
    ET.SubElement(world, "light", name="mapping_key_light", pos="0 -1 2", dir="0 .3 -1")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for n in HOME:
        ET.SubElement(
            actuator,
            "position",
            name="preview_" + n,
            joint=n,
            kp="10",
            kv=".5",
            forcelimited="true",
            forcerange="-1.7464 1.7464",
        )
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_view = visual.find("global")
    if global_view is None:
        global_view = ET.SubElement(visual, "global")
    global_view.attrib.update(offwidth="1280", offheight="960", azimuth="145", elevation="-25")
    path = ROOT / "mapping" / "robot_preview.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(path))


class Robot:
    def __init__(self, model):
        self.m = model
        self.d = mujoco.MjData(model)
        self.names = list(HOME)
        self.jids = np.array([model.joint(n).id for n in self.names])
        self.adr = model.jnt_qposadr[self.jids]
        self.dof = model.jnt_dofadr[self.jids]
        self.home = np.array(list(HOME.values()))
        self.low = model.jnt_range[self.jids, 0] + 0.025
        self.high = model.jnt_range[self.jids, 1] - 0.025
        self.sites = [model.site(n).id for n in LEGS]
        self.qhome = np.r_[0.0, 0.0, STAND_Z, 1.0, 0.0, 0.0, 0.0, self.home]
        self.d.qpos[:] = self.qhome
        mujoco.mj_forward(model, self.d)
        self.foot_home = self.d.site_xpos[self.sites].copy()
        self.home_z = self.foot_home[:, 2].copy()
        self.groups = [
            [i for i, n in enumerate(self.names) if n.startswith(leg) and "_J4_" not in n]
            for leg in LEGS
        ]
        self.jac = np.zeros((3, model.nv))
        self.rotjac = np.zeros_like(self.jac)
        self.bodyjac = np.zeros_like(self.jac)
        self.step_low = self.low.copy()
        self.step_high = self.high.copy()

    def pose(self, xyz, rpy, q):
        self.d.qpos[:3] = xyz
        quat = rotation(rpy).as_quat()
        self.d.qpos[3:7] = quat[[3, 0, 1, 2]]
        self.d.qpos[self.adr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def solve(self, xyz, rpy, target, q):
        q = q.copy()
        for leg, (sid, cols) in enumerate(zip(self.sites, self.groups)):
            previous = q[cols].copy()
            elbow_id = self.m.body(LEGS[leg] + "_forearm_link").id if leg < 2 else None

            def residual(x):
                q[cols] = x
                self.pose(xyz, rpy, q)
                err = np.r_[
                    self.d.site_xpos[sid] - target[leg],
                    0.001 * (x - self.home[cols]),
                    0.001 * (x - previous),
                ]
                if elbow_id is not None:
                    clearance = (
                        xyz[0] + PARAMS["front_elbow_clearance_x_m"] - self.d.xpos[elbow_id, 0]
                    )
                    err = np.r_[err, 2.0 * max(clearance, 0.0)]
                return err

            def jacobian(x):
                q[cols] = x
                self.pose(xyz, rpy, q)
                mujoco.mj_jacSite(self.m, self.d, self.jac, self.rotjac, sid)
                jac = self.jac[:, self.dof[cols]]
                out = np.vstack([jac, 0.001 * np.eye(len(cols)), 0.001 * np.eye(len(cols))])
                if elbow_id is not None:
                    clearance = (
                        xyz[0] + PARAMS["front_elbow_clearance_x_m"] - self.d.xpos[elbow_id, 0]
                    )
                    mujoco.mj_jacBody(self.m, self.d, self.bodyjac, self.rotjac, elbow_id)
                    row = (
                        -2.0 * self.bodyjac[0, self.dof[cols]]
                        if clearance > 0
                        else np.zeros(len(cols))
                    )
                    out = np.vstack([out, row])
                return out

            result = least_squares(
                residual,
                np.clip(previous, self.step_low[cols] + 1e-9, self.step_high[cols] - 1e-9),
                jac=jacobian,
                bounds=(self.step_low[cols], self.step_high[cols]),
                max_nfev=35,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            q[cols] = result.x
        self.pose(xyz, rpy, q)
        return q


def inside_polygon(com, feet, margin):
    hull = ConvexHull(feet)
    point = com.copy()
    # Project onto the half-space intersection, inset from each support edge.
    for _ in range(12):
        changed = False
        for eq in hull.equations:
            violation = eq[:2] @ point + eq[2] + margin
            if violation > 0:
                point -= violation * eq[:2]
                changed = True
        if not changed:
            break
    return point


def build_one(robot, name, count, offset):
    raw, meta = features(name, count)
    duration = (count - 1) / 30 * PARAMS["playback_time_scale"]
    times = np.arange(0, duration + DT / 2, DT)
    source_times = times / PARAMS["playback_time_scale"]
    source_times = np.clip(source_times, 0, (count - 1) / 30)
    src = np.arange(count) / 30
    f = {k: PchipInterpolator(src, v)(source_times) for k, v in raw.items()}
    n = len(times)
    poses = []
    actual = []
    targets = []
    errors = []
    margins = []
    self_pen = []
    floor_pen = []
    q = robot.home.copy()
    for k in range(n):
        robot.step_low = np.maximum(robot.low, q - PARAMS["max_joint_speed_rad_s"] * DT)
        robot.step_high = np.minimum(robot.high, q + PARAMS["max_joint_speed_rad_s"] * DT)
        rpy = np.array([f["roll"][k], f["pitch"][k], f["yaw"][k]])
        arms = np.array([f["left_hand_lift"][k], f["right_hand_lift"][k]])
        activation = np.clip(np.max(arms) / 0.035, 0, 1)
        xyz = np.array([-PARAMS["support_backshift_m"] * activation, f["body_y"][k], STAND_Z])
        target = robot.foot_home.copy()
        # Human leg timing reaches all six limbs. Front-limb gait contribution
        # fades out when human arm gestures require the front pair in the air.
        for j, leg in enumerate(LEGS):
            side = "left" if leg.startswith("L") else "right"
            delay = 0 if j < 4 else 0.22 * PARAMS["playback_time_scale"]
            kk = max(0, k - round(delay / DT))
            leg_y = f[side + "_foot_y"][kk]
            leg_z = f[side + "_foot_lift"][kk]
            weight = (1 - activation) if j < 2 else 1.0
            target[j, 1] += weight * leg_y
            target[j, 2] += weight * leg_z
        # The robot's arms are forward-facing; observed human hand lift and
        # outward movement define endpoint goals, not copied human joint angles.
        for j, side in enumerate(["left", "right"]):
            sign = 1 if side == "left" else -1
            target[j, 0] += (
                xyz[0] * activation
                + 0.018 * activation
                - 0.005 * np.clip(f[side + "_elbow_bend"][k], 0, 2)
            )
            target[j, 1] += xyz[1] * activation + sign * f[side + "_hand_out"][k] * activation
            target[j, 2] += arms[j]
        q = robot.solve(xyz, rpy, target, q)
        # Ground-near feet count as support; small adapted steps keep at least
        # the four rear/middle feet close to the ground. No aerial kick is copied.
        support = target[:, 2] - robot.home_z < 0.0045
        if support.sum() < 3:
            support[np.argsort(target[:, 2] - robot.home_z)[:3]] = True
        for _ in range(3):
            com = robot.d.subtree_com[robot.m.body("base_link").id, :2].copy()
            corrected = inside_polygon(com, target[support, :2], PARAMS["support_margin_m"])
            delta = corrected - com
            if np.linalg.norm(delta) < 0.0003:
                break
            xyz[:2] += delta
            q = robot.solve(xyz, rpy, target, q)
        # Report physical contacts for prescribed poses; this does not integrate
        # dynamics. The floor can resolve small geometric pad/site discrepancies.
        floor_id = robot.m.geom("mapping_floor").id
        # Foot pads are meshes, not points: orienting a foot changes its lowest
        # surface relative to its named site. Correct each penetrating limb's
        # endpoint upward, retaining the body target and the joint speed bound.
        for correction in range(4):
            mujoco.mj_forward(robot.m, robot.d)
            raise_by = np.zeros(6)
            for c in robot.d.contact:
                if floor_id not in (c.geom1, c.geom2) or c.dist >= -0.00015:
                    continue
                gid = c.geom2 if c.geom1 == floor_id else c.geom1
                gname = robot.m.geom(gid).name
                for j, leg in enumerate(LEGS):
                    if gname.startswith(leg + "_"):
                        raise_by[j] = max(raise_by[j], -float(c.dist) + 0.00015)
            if raise_by.max() < 0.0002:
                break
            target[:, 2] += raise_by
            q = robot.solve(xyz, rpy, target, q)
        mujoco.mj_forward(robot.m, robot.d)
        fp = 0.0
        sp = 0.0
        for c in robot.d.contact:
            if c.dist < 0:
                if floor_id in (c.geom1, c.geom2):
                    fp = max(fp, -float(c.dist))
                else:
                    sp = max(sp, -float(c.dist))
        errors.append(np.linalg.norm(robot.d.site_xpos[robot.sites] - target, axis=1))
        actual.append(robot.d.site_xpos[robot.sites].copy())
        targets.append(target.copy())
        floor_pen.append(fp)
        self_pen.append(sp)
        com = robot.d.subtree_com[robot.m.body("base_link").id, :2]
        hull = ConvexHull(target[support, :2])
        margins.append(float(np.min(-(hull.equations[:, :2] @ com + hull.equations[:, 2]))))
        poses.append(robot.d.qpos.copy())
    poses = np.array(poses)
    actual = np.array(actual)
    targets = np.array(targets)

    # Smooth entry and exit with all root and joint channels moving together.
    # Re-solving feet is intentionally not claimed for these blending sections.
    def blend(a, b, seconds):
        b = b.copy()
        if np.dot(a[3:7], b[3:7]) < 0:
            b[3:7] *= -1
        u = np.linspace(0, 1, round(seconds / DT) + 1)
        s = u * u * (3 - 2 * u)
        out = a[None] + s[:, None] * (b - a)[None]
        # Same-hemisphere quaternion interpolation followed by normalization.
        out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
        return out

    lead = blend(robot.qhome, poses[0], 1.0)[:-1]
    tail = blend(poses[-1], robot.qhome, 1.0)[1:]
    hold = np.repeat(robot.qhome[None], round(0.5 / DT), axis=0)
    full = np.concatenate([lead, poses, tail, hold])
    full_time = np.arange(len(full)) * DT
    full_rpy = Rotation.from_quat(full[:, [4, 5, 6, 3]]).as_euler("xyz")
    vel = np.zeros((len(full), robot.m.nv))
    for k in range(len(full) - 1):
        mujoco.mj_differentiatePos(robot.m, vel[k], DT, full[k], full[k + 1])
    q = full[:, robot.adr]
    source_time = np.full(len(full), np.nan)
    source_time[len(lead) : len(lead) + n] = offset + source_times
    detailed = ROOT / "mapping" / f"{name}_robot_trajectory.npz"
    np.savez_compressed(
        detailed,
        time=full_time.astype("f4"),
        qpos=full.astype("f4"),
        qvel=vel.astype("f4"),
        joint_names=np.array(robot.names),
        joint_pos=q.astype("f4"),
        source_time=source_time.astype("f4"),
        motion_start_index=len(lead),
        motion_end_index=len(lead) + n,
        feet_target=targets.astype("f4"),
        feet_actual=actual.astype("f4"),
        dt=np.float32(DT),
        reference_kind=np.array("kinematic draft; no trained policy"),
    )
    dest = ROOT / "trajectories" / name
    dest.mkdir(parents=True, exist_ok=True)
    channels = dict(
        time=full_time.astype("f4"),
        dt=np.float64(DT),
        phase=np.r_[np.zeros(len(lead)), np.ones(n), np.full(len(tail) + len(hold), 2)].astype(
            "f4"
        ),
        provenance_json=np.array(
            json.dumps(
                dict(
                    meta,
                    reference_kind="kinematic",
                    measured_channel_note="meas_* are schema aliases of kinematically solved states, not physics measurements",
                    parameters=PARAMS,
                )
            )
        ),
    )
    for prefix in ["", "meas_"]:
        for j, a in enumerate(["x", "y", "z"]):
            channels[prefix + "body_" + a] = full[:, j].astype("f4")
        for j, a in enumerate(["roll", "pitch", "yaw"]):
            channels[prefix + "body_" + a] = full_rpy[:, j].astype("f4")
        for leg, names in LEG_JOINTS.items():
            for j, jname in enumerate(names):
                channels[f"{prefix}L{leg}_j{j}"] = q[:, robot.names.index(jname)].astype("f4")
    reference = dest / "reference.npz"
    np.savez_compressed(reference, **channels)
    with (ROOT / "mapping" / f"{name}_joints.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *robot.names])
        writer.writerows(np.column_stack([full_time, q]))
    conversion = {"status": "not_checked"}
    try:
        path = ensure_motion_npz(DT, dest, reference="measured")
        conversion = dict(
            status="passed",
            path=str(path),
            note="schema and kinematic checks only; not dynamic feasibility",
        )
    except Exception as error:
        conversion = dict(status="failed", error=str(error))
    report = dict(
        name=name,
        source=meta,
        frames=len(full),
        dt=DT,
        duration_s=(len(full) - 1) * DT,
        source_mapping_window_s=[offset, offset + count / 30],
        source_mapping_window_semantics="start-inclusive end-exclusive",
        last_observed_source_frame_s=offset + (count - 1) / 30,
        reference=str(reference),
        detailed_trajectory=str(detailed),
        reference_type="kinematic candidate, no learning and no dynamic rollout",
        max_joint_speed_rad_s=float(np.abs(vel[:, robot.dof]).max()),
        min_joint_limit_margin_rad=float(
            np.min(
                np.minimum(
                    q - robot.m.jnt_range[robot.jids, 0], robot.m.jnt_range[robot.jids, 1] - q
                )
            )
        ),
        max_foot_target_error_m=float(np.max(errors)),
        mean_foot_target_error_m=float(np.mean(errors)),
        min_target_near_ground_polygon_margin_m=float(min(margins)),
        polygon_margin_note="geometric target-foot heuristic only, not a contact-based stability certificate",
        max_floor_penetration_m=float(max(floor_pen)),
        max_self_penetration_m=float(max(self_pen)),
        adaptation="arm priority; front gait attenuated while gesturing; rear/middle lift capped 3.5mm; COM shifted into support polygon",
        training_format_conversion=conversion,
    )
    # Audit the complete reference including lead-in/out and final hold. This is
    # independent of the target-foot heuristic used to design the main motion.
    whole_floor = whole_self = 0.0
    actual_contact_counts = []
    worst_pairs = {}
    actual_contact_margins = []
    contact_margins_missing = 0
    for frame, row in enumerate(full):
        robot.d.qpos[:] = row
        mujoco.mj_forward(robot.m, robot.d)
        contacts = []
        for c in robot.d.contact:
            pair = [robot.m.geom(c.geom1).name, robot.m.geom(c.geom2).name]
            if floor_id in (c.geom1, c.geom2):
                whole_floor = max(whole_floor, max(0.0, -float(c.dist)))
                if c.dist <= 0.0005:
                    contacts.append(np.array(c.pos[:2]))
            else:
                whole_self = max(whole_self, max(0.0, -float(c.dist)))
            if c.dist < -0.0002:
                key = " | ".join(sorted(pair))
                old = worst_pairs.get(key)
                if old is None or -c.dist > old["penetration_m"]:
                    worst_pairs[key] = dict(penetration_m=-float(c.dist), time_s=frame * DT)
        actual_contact_counts.append(len(contacts))
        if len(contacts) >= 3:
            try:
                hull = ConvexHull(np.asarray(contacts))
                com = robot.d.subtree_com[robot.m.body("base_link").id, :2]
                actual_contact_margins.append(
                    float(np.min(-(hull.equations[:, :2] @ com + hull.equations[:, 2])))
                )
            except Exception:
                contact_margins_missing += 1
        else:
            contact_margins_missing += 1
    report["full_reference_audit"] = dict(
        frames=len(full),
        includes_lead_and_tail=True,
        max_floor_penetration_m=whole_floor,
        max_self_penetration_m=whole_self,
        min_geometric_floor_contacts=min(actual_contact_counts),
        min_contact_polygon_margin_m=min(actual_contact_margins)
        if actual_contact_margins
        else None,
        frames_without_nondegenerate_contact_polygon=contact_margins_missing,
        note="contacts on prescribed poses; no contact-force or dynamic stability guarantee",
        worst_penetration_pairs=worst_pairs,
    )
    (ROOT / "mapping" / f"{name}_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return full, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    model = make_model()
    robot = Robot(model)
    rules = dict(
        schema="jumper-human-retarget/1",
        parameters=PARAMS,
        human_sides="MediaPipe anatomical left/right; no mirroring based on screen position",
        mapping=[
            dict(
                human="left shoulder/elbow/wrist",
                robot="LF_J0..J3",
                method="left front endpoint IK",
            ),
            dict(
                human="right shoulder/elbow/wrist",
                robot="RF_J0..J3",
                method="right front endpoint IK",
            ),
            dict(
                human="left hip/knee/ankle",
                robot=["LF", "LM", "LR"],
                method="adapted left step features; rear phase delay; front-arm priority",
            ),
            dict(
                human="right hip/knee/ankle",
                robot=["RF", "RM", "RR"],
                method="adapted right step features; rear phase delay; front-arm priority",
            ),
            dict(
                human="shoulder/hip centers",
                robot="free base",
                method="scaled torso roll/pitch/yaw and lateral shift plus robot-specific support correction",
            ),
        ],
        grippers="LF_J4 and RF_J4 held at HOME=0; no human finger mapping",
        limitations=[
            "monocular 3D estimates are ambiguous",
            "human two-leg motion is adapted, not copied into six independent measured legs",
            "third clip maps only the 103-frame full-body prefix",
            "kinematic preview is not proof of physically executable control",
            "no policy trained",
        ],
    )
    (ROOT / "mapping" / "mapping_rules.json").write_text(json.dumps(rules, indent=2) + "\n")
    reports = []
    for name, count, offset in JOBS:
        full, report = build_one(robot, name, count, offset)
        if args.render:
            from jumper_mapping_renderer import render_reference

            report["preview"] = render_reference(
                model, full, DT, ROOT / "previews" / f"{name}_robot_reference.mp4"
            )
        reports.append(report)
    summary = ROOT / "mapping" / "validation_report.json"
    summary.write_text(
        json.dumps(
            dict(status="kinematic mapping draft", training_started=False, reports=reports),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
