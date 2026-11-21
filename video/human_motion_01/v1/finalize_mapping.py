"""Validate delivered motion artifacts and write English/Chinese documentation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
NAMES = ["01_side_step", "02_kick_step", "03_arm_body_sway"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=None, help="source footage matching the extraction manifest"
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    source = args.source or Path(manifest["source"])
    if not source.is_absolute():
        source = ROOT.parents[2] / source
    if not source.is_file():
        parser.error("Raw source footage is not distributed; provide --source to revalidate it")
    assert sha(source) == manifest["source_sha256"]
    from tasks.jumper.common.dance.motion import LEG_JOINTS

    checks = []
    for name in NAMES:
        with np.load(ROOT / "mapping" / f"{name}_robot_trajectory.npz") as z:
            names = z["joint_names"].tolist()
            planned = z["joint_pos"]
            count = len(planned)
        with np.load(ROOT / "mapping" / f"{name}_pd_rollout.npz") as z:
            actual = z["qpos"]
        reference = ROOT / "trajectories" / name / "reference.npz"
        with np.load(reference) as z:
            assert abs(float(z["dt"]) - 0.02) < 1e-12
            assert len(z["time"]) == len(actual) == count
            for leg, joints in LEG_JOINTS.items():
                for j, n in enumerate(joints):
                    col = names.index(n)
                    assert np.allclose(z[f"L{leg}_j{j}"], planned[:, col], atol=1e-7)
                    assert np.allclose(z[f"meas_L{leg}_j{j}"], actual[:, 7 + col], atol=1e-7)
            assert np.allclose(z["meas_body_x"], actual[:, 0], atol=1e-7)
            assert np.allclose(z["meas_body_y"], actual[:, 1], atol=1e-7)
            assert np.isfinite(np.column_stack([z[f"meas_L0_j{j}"] for j in range(5)])).all()
            assert np.max(np.abs(z["L0_j4"])) == 0 and np.max(np.abs(z["L5_j4"])) == 0
            assert "physically measured" in json.loads(str(z["provenance_json"]))["reference_kind"]
        with np.load(reference.parent / ".cache/reference.motion.npz") as z:
            assert len(z["joint_pos"]) == count
            assert len(z["body_names"]) == z["body_pos_w"].shape[1]
            assert all(
                np.isfinite(z[k]).all()
                for k in [
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ]
            )
        checks.append(
            dict(
                name=name,
                samples=count,
                reference_sha256=sha(reference),
                checks="planned/measured channels match separate trajectories; joint order and timebase agree; converted arrays finite and full-length",
            )
        )
    for script in ROOT.glob("*.py"):
        ast.parse(script.read_text(), filename=str(script))
    physics = json.loads((ROOT / "mapping/physics_probe.json").read_text())
    assert all(r["first_fall_time_s"] is None for r in physics)
    assert all(r["measured_reference_format_check"] == "passed" for r in physics)
    report = dict(
        status="completed mapping and bounded preview checks; no RL training",
        source_sha256=manifest["source_sha256"],
        source_unchanged=True,
        robot_xml_sha256=sha(REPO / "assets/jumper/jumper.xml"),
        model="canonical Jumper with extra floor and simple torque-limited position actuators in preview only",
        training_started=False,
        training_task_registered=False,
        primary_reference="trajectories/<motion>/reference.npz; measured channels from free-base PD physics",
        kinematic_stage_report="mapping/validation_report.json (before physical refinement)",
        physics_stage_report="mapping/physics_probe.json",
        checks=checks,
    )
    (ROOT / "mapping/delivery_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    table = "\n".join(
        f"| {r['name']} | {r['steps']} | {r['duration_s']:.2f} | {r['joint_rmse_rad']:.4f} | {r['root_xy_rmse_m'] * 1000:.1f} |"
        for r in physics
    )
    english = f"""# Human-to-Jumper mapping, first version

Three video-derived motion mappings have been generated. No reinforcement
learning, policy optimization or new training-task registration was performed.
The original video, training code and existing checkpoints are unchanged.

## Inspect these files first

- `previews/*_comparison.mp4`: left = human video slowed 1.8x; right = robot
  free-base simple-PD physics playback. One-second lead-in and 1.5-second final
  transition/hold are included. These are not learned-policy demonstrations.
- `mapping/mapping_rules.json`: human-to-robot rules and numeric adaptations.
- `trajectories/<motion>/reference.npz`: primary reference in the existing dance
  loader's source format. Commands are mapped targets; `meas_*` are states
  actually reached in the bounded MuJoCo PD probe.
- `mapping/delivery_manifest.json`: final checks and reference hashes.

## Mapping decisions

MediaPipe Pose Landmarker 1.2.0 identified 33 landmarks per frame. Observations,
estimated world coordinates and visibility scores are in `poses/<motion>/`.
These monocular estimates are not ground-truth motion capture. The model and
isolated helper packages are in `.tools/`; the Conda training environment was
not modified. Model provenance is in each pose `metadata.json`.

| Human feature | Jumper mapping |
| --- | --- |
| Left shoulder/elbow/wrist | Left front arm `LF_J0..J3`, endpoint inverse kinematics |
| Right shoulder/elbow/wrist | Right front arm `RF_J0..J3`, endpoint inverse kinematics |
| Left hip/knee/ankle | Adapted stepping features for LF/LM/LR |
| Right hip/knee/ankle | Adapted stepping features for RF/RM/RR |
| Shoulder/hip centers and estimated torso orientation | Scaled free-base translation and roll/pitch/yaw, plus support-aware backward shift |
| Fingers | No human finger reconstruction; both gripper command targets remain at HOME=0 |

Anatomical landmark labels determine left and right; screen halves are not used
to guess them. The front pair are both arms and legs: arm gestures take priority,
and their walking contribution fades while they are raised. Middle/rear legs
carry an adapted gait, with a 0.22-second source-time delay for the rear pair.
This is not a one-to-one transfer of human two-legged locomotion. Rear/middle
foot lift is capped at 3.5 mm, lateral foot displacement at 9 mm, and mapped joint
speed at 4.5 rad/s. Large kicks become conservative foot/stance movements.

For the third motion, only source 24.900--28.333 seconds (103 frames, before the
close-view cut) is mapped. The full extracted third video is still preserved.
All durations are retimed by 1.8, then entry/exit and standing hold are added.

## Reference and playback checks

| Motion | Samples | Playback seconds | PD joint RMSE rad | Root XY RMSE mm |
| --- | --- | --- | --- | --- |
{table}

All three bounded free-base PD probes had no fall under the recorded criterion
(base height below 55 mm or body-up cosine below 0.65). They use gravity and
resolved contacts, no root forces, kp=10 / kv=0.5 and a 1.7464 Nm torque limit.
They do not use the project's complete torque-speed servo model and do not prove
hardware execution, robust policy performance or full imitation accuracy.

`mapping/validation_report.json` audits the *planned* kinematic draft, including
lead-in/out, and records remaining target errors and geometry penetrations.
The largest planned pad/floor overlap is about 2.3 mm; no self-penetration was
observed in that prescribed full-reference audit. A near-ground target polygon
is only a design heuristic, not a stability certificate: many prescribed poses
have too few actual contacts. The subsequent physics probe resolves contacts
and supplies the measured reference channels. Root drift and tracking errors
remain; see `mapping/physics_probe.json` for the measured values.

Each primary reference passes `ensure_motion_npz(.02, directory,
reference="measured")`; converted MotionLoader data are in each `.cache/`.
The cached conversion's sample count matches the full measured trajectory.
This is data preparation, not a registered or trained new behavior.

## Files and reproduction

- `clips/`: trimmed source videos, unchanged by mapping.
- `poses/`: observed 33-point landmark arrays, visibility and feature CSVs.
- `mapping/*_robot_trajectory.npz`: planned qpos/qvel and requested/achieved IK foot points.
- `mapping/*_joints.csv`: human-readable planned robot joint angles in radians.
- `mapping/*_kinematic_source.npz`: schema-form draft before physical refinement.
- `mapping/*_pd_rollout.npz`: states reached in the simple-PD physics probe.
- `trajectories/`: one independent primary reference per motion.
- `previews/*_human_pose.mp4`: observed human landmark overlays.
- `previews/*_robot_reference.mp4`: prescribed kinematic target previews.
- `previews/*_pd_physics.mp4`: recorded physical playback, without learning.

Interactive replay (change `--motion` to 2 or 3):

```sh
conda activate pytorch
cd jumper-human-motion
python video/human_motion_01/v1/preview_mapping.py --motion 1 --loop
```

Default mode replays the recorded physical probe. `--mode planned` displays the
prescribed kinematic target instead; neither mode trains or executes a learned policy.

Rebuild mapping and playback, using the existing extracted poses:

```sh
cd jumper-human-motion
MUJOCO_GL=egl python video/human_motion_01/v1/build_motion_mapping.py --render
python video/human_motion_01/v1/check_mapping_physics.py
MUJOCO_GL=egl python video/human_motion_01/v1/create_mapping_previews.py
python video/human_motion_01/v1/finalize_mapping.py
```

The order matters: the mapping builder writes a kinematic draft; the physics
probe replaces `meas_*` with simulated observations. To extract poses again:

```sh
PYTHONPATH=video/human_motion_01/v1/.tools/pose_packages python video/human_motion_01/v1/extract_human_pose.py
```

Pose tool reference: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
"""
    (ROOT / "MAPPING.md").write_text(english)
    chinese = f"""# 人体到 Jumper 的映射说明（第一版）

已完成三个动作的人体关键点提取、机器人映射、参考轨迹和预览。没有启动强化学习，没有注册新的训练任务，也没有修改原视频、原训练代码或已训练的权重。

## 先看哪些文件

- `previews/01_side_step_comparison.mp4`：闪身步对照。
- `previews/02_kick_step_comparison.mp4`：踢腿与交叉步对照。
- `previews/03_arm_body_sway_comparison.mp4`：张臂摆身对照。
- 对照视频左边是放慢到原速度约 1/1.8 的人体动作，右边是没有训练、仅用 PD 电机跟随关节目标的物理回放；额外加入了入场、收尾和站立衔接。
- `mapping/mapping_rules.json`：人体到机器人各部位的映射规则、幅度和速度参数。
- `trajectories/动作名/reference.npz`：后续训练可读取的参考动作数据。
- `mapping/delivery_manifest.json`：最终检查清单和文件校验值。

## 映射方法

先用 MediaPipe Pose Landmarker 1.2.0 逐帧提取 33 个人体关键点，保存坐标和可见性。这是普通单目视频的估计，不是精确动作捕捉。辅助依赖和模型均隔离放在 `.tools/`，未修改原 Conda 训练环境。

| 人的部位 | Jumper 对应部位 |
| --- | --- |
| 左肩、左肘、左腕 | 左前肢 LF，通过足端/手端目标求解关节角度 |
| 右肩、右肘、右腕 | 右前肢 RF，通过足端/手端目标求解关节角度 |
| 左髋、左膝、左踝 | LF、LM、LR 的改编迈步特征 |
| 右髋、右膝、右踝 | RF、RM、RR 的改编迈步特征 |
| 肩部和髋部中心、躯干朝向 | 机身平移和倾斜/转向，并加入机器人支撑所需的后移 |
| 手指 | 没有重建手指；两只前爪的关节命令保持 HOME=0 |

左右根据人体识别模型的解剖学标签区分，不按屏幕左右猜测。前肢同时承担手臂和腿的功能：抬前肢表演时，优先执行手臂动作、减弱该前肢的迈步贡献，其余四足提供支撑。后足比同侧中足延后约 0.22 秒原动作时间。

为了适配机器人，后/中足抬脚限制为 3.5 毫米，横向足端变化限制为 9 毫米，映射后的关节速度限制为 4.5 弧度/秒。人的大幅踢腿被改编成较小幅度的抬脚和站姿变化，不是直接复制成人类的双足踢击。

第三个动作只映射原视频 24.900～28.333 秒、全身可见的 103 帧，没有把近景中不可见的腿脚当成可靠数据。完整第三段视频仍保留。

## 数据检查与物理回放

| 动作 | 样本数 | 回放秒数 | PD 关节均方根误差（弧度） | 机身 XY 均方根误差（毫米） |
| --- | --- | --- | --- | --- |
{table}

三个短时自由机身物理回放都没有触发摔倒条件：机身低于 55 毫米或朝上的余弦小于 0.65。开启了重力和接触，没有给机身额外支撑力；使用 kp=10、kv=0.5、力矩限制 1.7464 牛米的简单 PD 电机。

这不是完整舵机速度—力矩模型，也不是已经学会动作的策略。跟踪误差和机身漂移仍存在，实际数值在 `mapping/physics_probe.json` 中。

`mapping/validation_report.json` 记录计划轨迹的检查，包含入场和收尾。计划轨迹最大脚垫/地面重叠约 2.3 毫米，全片未观察到自身穿透。近地目标点形成的支撑多边形只是设计启发式，不能当成实际稳定性保证；很多指定姿态并没有足够的真实接触点。因此，又进行了接触解算的物理回放，并将实际达到的姿态写入参考数据。

`reference.npz` 中，不带 `meas_` 的字段保存映射目标；`meas_*` 字段保存真实 MuJoCo 物理回放的姿态。三个参考都通过了项目现有的动作读取/转换检查，转换结果位于各自 `.cache/` 中，样本数与完整轨迹一致。当前完成的是数据准备，没有注册或训练新的行为。

## 文件夹与使用方法

- `clips/`：原来裁剪的视频，没有被映射步骤修改。
- `poses/`：人体关键点、可见性、动作特征 CSV；人体识别预览在 `previews/*_human_pose.mp4`。
- `mapping/`：映射规则、计划关节角度 CSV、详细轨迹、物理回放记录和检查结果。
- `trajectories/`：每个动作一个独立目录，里面的 `reference.npz` 是当前主要参考数据。
- `previews/*_robot_reference.mp4`：直接指定姿态的运动学预览。
- `previews/*_pd_physics.mp4`：没有训练、仅用电机跟随的物理回放。

在 MuJoCo 窗口中循环查看第一个动作：

```sh
conda activate pytorch
cd jumper-human-motion
python video/human_motion_01/v1/preview_mapping.py --motion 1 --loop
```

把 `--motion 1` 改为 `2` 或 `3` 可以看另外两个动作。默认重放已记录的物理结果；增加 `--mode planned` 可以看原计划轨迹。两种模式都不训练、不执行学习策略。

如需重新生成，依次执行：

```sh
cd jumper-human-motion
MUJOCO_GL=egl python video/human_motion_01/v1/build_motion_mapping.py --render
python video/human_motion_01/v1/check_mapping_physics.py
MUJOCO_GL=egl python video/human_motion_01/v1/create_mapping_previews.py
python video/human_motion_01/v1/finalize_mapping.py
```

顺序不能颠倒：映射脚本先生成运动学目标，物理检查脚本再将实际回放姿态写入 `meas_*` 通道。

人体识别工具说明：https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
"""
    h = sha(ROOT / "MAPPING.md")
    (ROOT / "MAPPING.zh.md").write_text(
        f"<!-- tracks: MAPPING.md @ sha256:{h[:16]} -->\n\n" + chinese
    )
    extraction = ROOT / "README.md"
    text = extraction.read_text()
    old = "This package contains video extraction only. No human keypoints, robot joint\ntrajectories, pose retargeting or training have been performed."
    new = "This document describes the original video-extraction stage. Human landmark\nextraction, retargeting, references and previews are now documented in\n[MAPPING.md](MAPPING.md). No reinforcement learning has been started."
    extraction.write_text(text.replace(old, new))
    translation = ROOT / "README.zh.md"
    zh = translation.read_text()
    zh = zh.replace(
        "当前只完成视频片段提取，没有提取人体关键点、制作机器人轨迹、执行动作映射或启动训练。",
        "本文记录最初的视频片段提取。现已完成的人体关键点、机器人映射、参考轨迹和预览见 [映射说明](MAPPING.zh.md)。尚未启动强化学习训练。",
    )
    zh = zh.replace(
        "动作时间由人工查看画面选定，尚未自动识别人体姿态。这些是后续映射使用的视频素材，不能直接作为关节轨迹交给 PPO 训练。",
        "动作时间由人工查看画面选定。这里描述的视频片段本身不能直接作为关节轨迹交给 PPO 训练；后续映射生成的数据和人体识别结果另见映射说明。",
    )
    zh = zh.split("\n", 1)[1].lstrip("\n")
    translation.write_text(f"<!-- tracks: README.md @ sha256:{sha(extraction)[:16]} -->\n\n" + zh)
    (ROOT / ".gitignore").write_text(".tools/\n__pycache__/\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
