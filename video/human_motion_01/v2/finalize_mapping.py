"""Validate expressive motion, retain targets, and package measured references."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tasks.jumper.common.dance.motion import LEG_JOINTS, load_source, ensure_motion_npz

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "v1"
LEGS = ["LF", "RF", "LM", "RM", "LR", "RR"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=None, help="source footage matching the extraction manifest"
    )
    args = parser.parse_args()
    old = json.loads((SOURCE / "manifest.json").read_text())
    source = args.source or Path(old["source"])
    if not source.is_absolute():
        source = ROOT.parents[2] / source
    if not source.is_file():
        parser.error("Raw source footage is not distributed; provide --source to revalidate it")
    assert sha(source) == old["source_sha256"]
    for c in old["clips"]:
        clip = SOURCE / c["file"]
        if not clip.is_file():
            parser.error("Source clips are omitted; run v1/extract_clips.py --source first")
        assert sha(clip) == c["sha256"]
    json.loads((ROOT / "mapping/validation_report.json").read_text())
    physics = json.loads((ROOT / "mapping/physics_report.json").read_text())
    assert all(r["first_fall_time_s"] is None for r in physics)
    byname = {r["name"]: r for r in physics}
    assert byname["01_side_step"]["root_xy_span_m"][1] > 0.06
    for name in ["01_side_step", "03_arm_body_sway"]:
        assert all(byname[name]["feet_world_span_m"][leg][2] > 0.025 for leg in LEGS[2:])
    assert byname["02_kick_step"]["feet_relative_body_span_m"]["RR"][0] > 0.13
    assert byname["02_kick_step"]["feet_relative_body_span_m"]["RR"][2] > 0.05
    model = mujoco.MjModel.from_xml_path(str(ROOT / "mapping/robot_preview.xml"))
    delivery = []
    for r in physics:
        name = r["name"]
        path = ROOT / "trajectories" / name / "reference.npz"
        with np.load(path) as z:
            channels = {k: z[k].copy() for k in z.files}
        with np.load(ROOT / "mapping" / f"{name}_pd_rollout.npz") as z:
            actual = z["qpos"].astype(float)
        # Make the planned-format copy deterministic even if this finalizer is
        # re-run after measured channels have already been written.
        draft = {k: v.copy() for k, v in channels.items()}
        for k in list(draft):
            if k.startswith("meas_"):
                draft[k] = draft[k[5:]].copy()
        draft_provenance = json.loads(str(draft["provenance_json"]))
        draft_provenance.pop("physics", None)
        draft_provenance["reference_kind"] = (
            "planned expressive kinematic candidate; no trained policy"
        )
        draft_provenance["measured_alias_note"] = (
            "In this planned copy, meas_* are aliases of command channels, not physical measurements."
        )
        draft["provenance_json"] = np.array(json.dumps(draft_provenance))
        np.savez_compressed(ROOT / "mapping" / f"{name}_planned_reference.npz", **draft)
        rpy = Rotation.from_quat(actual[:, [4, 5, 6, 3]]).as_euler("xyz")
        for j, a in enumerate(["x", "y", "z"]):
            channels["meas_body_" + a] = actual[:, j].astype("f4")
        for j, a in enumerate(["roll", "pitch", "yaw"]):
            channels["meas_body_" + a] = rpy[:, j].astype("f4")
        for leg, names in LEG_JOINTS.items():
            for j, n in enumerate(names):
                channels[f"meas_L{leg}_j{j}"] = actual[:, model.joint(n).qposadr[0]].astype("f4")
        provenance = json.loads(str(channels["provenance_json"]))
        provenance["reference_kind"] = (
            "expressive free-base MuJoCo PD measured candidate; no trained policy"
        )
        provenance["measured_alias_note"] = (
            "meas_* now contain actual physics states, not aliases; commanded channels still preserve the planned expressive target"
        )
        provenance["physics"] = (
            "gravity and contacts, simple kp10 kv.5 torque1.7464Nm, no root forces"
        )
        channels["provenance_json"] = np.array(json.dumps(provenance))
        np.savez_compressed(path, **channels)
        loaded = load_source(path, reference="measured")
        assert len(loaded.joint_pos) == len(actual)
        assert np.isfinite(loaded.joint_pos).all() and np.isfinite(loaded.root_pos).all()
        try:
            converted = ensure_motion_npz(0.02, path.parent, reference="measured")
            compatibility = dict(status="passed", path=str(converted))
        except ValueError as error:
            compatibility = dict(
                status="requires_a_task_with_scheduled_contacts",
                reason=str(error),
                note="The existing dance converter assumes all four legs stay grounded; stepping and kicking intentionally violate that assumption.",
            )
        delivery.append(
            dict(
                name=name,
                reference=str(path),
                sha256=sha(path),
                source_schema_read="passed",
                fixed_support_dance_compatibility=compatibility,
            )
        )
    for script in ROOT.glob("*.py"):
        ast.parse(script.read_text(), filename=str(script))
    manifest = dict(
        version=2,
        status="expressive mappings and bounded physical previews completed",
        source_unchanged=True,
        training_started=False,
        registered_training_task=False,
        mapping="hands to LF/RF only; feet to LM/LR and RM/RR only; shared root balance compensation",
        method="video features with manually adapted four-leg contact timing and reviewed right-kick phases",
        motion_acceptance=dict(
            side_step_body_y_over_60mm=True,
            all_walking_feet_lift_over_25mm=True,
            right_kick_x_over_130mm=True,
            right_kick_lift_over_50mm=True,
            no_fall_in_bounded_PD_probes=True,
        ),
        validation_scope="Not exact human-motion reconstruction, robust policy validation, or hardware validation",
        references=delivery,
    )
    (ROOT / "mapping/delivery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = "\n".join(
        f"| {r['name']} | {r['root_xy_span_m'][1] * 1000:.1f} | {r['feet_relative_body_span_m']['RR'][0] * 1000:.1f} | {r['feet_relative_body_span_m']['RR'][2] * 1000:.1f} | {r['joint_rmse_rad']:.4f} |"
        for r in physics
    )
    english = f"""# Expressive mapping v2

This revision uses human arm observations only for LF/RF and human steps only
for LM/LR and RM/RR. It preserves much larger foot trajectories and the reviewed
right-leg retract/forward-swing kick. No learning or training-task registration
was performed. Version 1 is preserved in the sibling `../v1/` directory.

## View the result

- `previews/*_comparison.mp4`: human versus prescribed mapping target.
- `previews/*_physics_comparison.mp4`: human versus actual free-base PD execution.
- Labels distinguish target animation from physical execution. No root support
  force, learned policy, or hidden assistance is used in physical execution.
- Camera 115 degrees shows the right-rear kick and the front arms clearly.

## Mapping and amplitude

| Human part | Robot part |
| --- | --- |
| Left arm | LF only |
| Right arm | RF only |
| Left foot/step | LM and LR |
| Right foot/step | RM and RR; RR performs the principal kick |
| Torso and balance | Shared root pose; planted-foot compensation |

The four legs use a wider 210 mm lateral stance. Walking clearance targets are
45 mm for the side steps and 55 mm for the body-sway steps. RR's kick retracts
to x=-200 mm then swings forward to x=-40 mm at z=70 mm, versus the 10 mm
standing foot-site height. The 160 mm fore/aft stroke is preserved. A 20 mm
outward displacement makes that height reachable without the old knee-stop
singularity. Front-limb targets never receive foot-step features.

The second clip mainly kicks with the human right leg. LM/LR therefore support
that performance rather than inventing simultaneous left-leg kicks. RM performs
the preparation and RR the large kick. Motions 1 and 3 visibly step with all four
legs. Shared body and balance changes can move all joints without changing this
semantic separation.

Source landmarks are reused from `../v1/poses/`. This is a video-feature
mapping with manually authored crawl contact order and reviewed kick phases,
not a frame-exact reconstruction of human two-legged motion. Clip 3 still uses
the first 103 full-body frames. Timing is slowed 2x for clips 1/2 and 2.5x for
clip 3; half-second initial/final holds are added. Clip 3's final steps land
before its final large arm gesture, preserving the lift while improving balance.

## Measured physical motion

| Clip | Body Y range mm | RR X range relative to body mm | RR Z range relative to body mm | Joint RMSE rad |
| --- | --- | --- | --- | --- |
{rows}

All three single bounded probes did not cross the fall criterion (base below
55 mm or body-up cosine below .65). The simulator uses gravity/contact and simple
kp10/kv.5 PD actuators with 1.7464 Nm torque limits. It is not the full servo
torque-speed model. Clip 3 reaches roughly 39 degrees of body tilt and retains
more tracking error; do not describe these runs as comprehensive stability tests.

The planned IK targets are intentionally expressive, and not all targets are
reachable exactly: the largest original target residuals are about 58, 44 and
105 mm. Planned pose geometry includes overlaps up to about 4.1 mm in clip 1.
These are recorded in `mapping/validation_report.json`, not silently marked as
passed. The actual PD records resolve contacts and differ from those targets;
both versions are retained so visual similarity and execution can be judged.

## Reference data

- `mapping/mapping_rules.json`: semantic rules and numerical settings.
- `mapping/*_robot_trajectory.npz`: achieved planned qpos, before dynamic execution.
- `mapping/*_pd_rollout.npz`: actual free-base PD execution states and foot paths.
- `mapping/*_planned_reference.npz`: source-format planned copy.
- `trajectories/<clip>/reference.npz`: planned command channels plus physically
  observed `meas_*` channels. The measured tracks preserve the large motion;
  they are not the old small-motion fallback. They are simulation measurements,
  not hardware measurements or learned-policy trajectories.
- `mapping/physics_report.json`: physical amplitude, body motion and fall results.
- `mapping/delivery_manifest.json`: amplitude gates, reference hashes and status.

Source-format loading passes for all three clips. The existing dance converter
accepts clip 1, but rejects clips 2 and 3 at its four-support-foot coplanarity
check. Passing that check alone does not validate stepping or kicking behavior.
These motions need a future task with scheduled contacts before training; do
not shrink them to pass an incompatible fixed-support check. No training
command is supplied here.

## Interactive viewing

```sh
conda activate pytorch
python video/human_motion_01/v2/preview_mapping.py --motion 2 --loop
```

Default is recorded physics; `--mode planned` views the prescribed target.
Change `--motion` to 1 or 3 for the other clips. Neither mode trains a policy.

## Reproduce

From `/home/jm/code/jumper` in the existing pytorch environment:

```sh
python video/human_motion_01/v2/build_mapping.py
python video/human_motion_01/v2/check_physics.py
MUJOCO_GL=egl python video/human_motion_01/v2/create_previews.py
python video/human_motion_01/v2/finalize_mapping.py
```

The builder writes expressive kinematic candidates; the finalizer retains them
and writes the physically observed measured channels without changing the motion
targets. Source videos, pose estimates, v1 data and existing model checkpoints
remain unchanged.
"""
    (ROOT / "README.md").write_text(english)
    zh = f"""# 第二版：大幅度手脚分离映射

这版将人的双臂只对应 LF/RF 两条前肢，将人的脚步只对应其余 LM、RM、LR、RR 四条腿。没有开始训练。第一版保留在同级的 `../v1/` 目录。

## 先看视频

- `previews/*_comparison.mp4`：左边人体，右边映射目标，属于指定姿态的预览。
- `previews/*_physics_comparison.mp4`：左边人体，右边实际电机跟踪的物理回放。
- 两种视频明确标注类别。物理回放开启重力和接触，没有给机身额外支撑力，没有学习策略。
- 相机改为 115 度偏侧视角，便于看清右后腿从后向前踢动。

## 这次改了什么

| 人的部位 | 螃蟹对应部位 |
| --- | --- |
| 左臂 | 只控制 LF 左前肢 |
| 右臂 | 只控制 RF 右前肢 |
| 左脚与左侧脚步 | LM、LR |
| 右脚与右侧脚步 | RM、RR，其中 RR 承担主要踢腿 |
| 躯干与平衡 | 共享机身姿态和支撑补偿 |

四腿采用更外展的站姿，基准脚点到机身中线的横向距离为 210 毫米。闪身步目标抬脚 45 毫米，摆身步目标抬脚 55 毫米。右后腿先向后收至 x=-200 毫米，再前摆至 x=-40 毫米，前后目标行程 160 毫米、抬高约 60 毫米，并外摆 20 毫米，避免在内收姿态碰到膝关节限位。前肢不再混入脚步特征。

第二段原视频主要是人的右腿在踢，所以左侧 LM/LR 保持支撑，不伪造左右同时踢腿。RM 做起势配合，RR 做主要摆踢。第一、三段则四条腿都有明显迈步。机身的共享平衡补偿会带动全身关节，这是身体协调，不是将脚步混入前肢表演。

方法是人体视频特征加人工改编的四足步序和踢腿阶段，不是逐帧精确还原人体双足动作。第三段仍使用 103 帧全身可见的部分。第一、二段放慢为原速度的 1/2，第三段为 1/2.5，并加入首尾各半秒停留。第三段让最后一组脚步在最后一次大幅摆臂前落地，保留抬脚幅度、调整时序。

## 实际物理回放测量

| 动作 | 机身左右行程（毫米） | RR 相对机身前后行程（毫米） | RR 相对机身上下行程（毫米） | 关节均方根误差（弧度） |
| --- | --- | --- | --- | --- |
{rows}

三个单次短回放都没有触发摔倒条件（机身低于 55 毫米，或朝上余弦低于 0.65）。使用 kp10、kv0.5、力矩限制 1.7464 牛米的简单 PD 电机，不是完整舵机速度—力矩模型，也不是已训练的策略。第三段仍有约 39 度的大倾斜和较大的跟踪偏差，不能把这些检查称为全面稳定性验证。

映射原始目标并非全部能精确达到：三个动作的最大原始目标残差约为 58、44、105 毫米，第一段指定姿态还存在约 4.1 毫米的几何重叠，详见 `mapping/validation_report.json`。没有把这些结果写成全部通过。物理回放会解算接触，并与目标有偏差；目标和实际记录分别保留，供判断相似度和可执行性。

## 文件在哪里

- `mapping/mapping_rules.json`：本版手脚对应关系、幅度和时间参数。
- `mapping/*_robot_trajectory.npz`：物理执行前的计划姿态轨迹。
- `mapping/*_pd_rollout.npz`：实际物理回放状态和足端轨迹。
- `mapping/*_planned_reference.npz`：原计划的参考格式副本。
- `trajectories/动作名/reference.npz`：不带 `meas_` 的字段是计划命令，`meas_*` 是实际物理回放记录。实际记录保留了大幅度动作，没有换回旧的小动作。这是仿真测量，不是真机测量或训练结果。
- `mapping/physics_report.json`：真实回放的幅度、机身移动、跟踪误差和摔倒检查。
- `mapping/delivery_manifest.json`：幅度检查门槛、文件校验值和状态。

三个参考都能按源数据格式读取。现有舞蹈转换器的四足共面检查中，第一段通过，第二、三段未通过；通过这一项也不代表已验证迈步和踢腿任务。后续需要建立支持动态接触时序的任务，才能接入训练。不能为了通过这个旧检查再把动作缩小。当前没有提供或执行训练命令。

## 在 MuJoCo 窗口查看

```sh
conda activate pytorch
python video/human_motion_01/v2/preview_mapping.py --motion 2 --loop
```

默认重放物理记录，增加 `--mode planned` 可查看映射目标。把动作编号改为 1 或 3 可切换。另外，上一级原来的预览入口也已默认指向第二版，使用 `--version 1` 仍可看旧版。

## 重新生成

在项目根目录、现有 pytorch 环境中依次执行：

```sh
python video/human_motion_01/v2/build_mapping.py
python video/human_motion_01/v2/check_physics.py
MUJOCO_GL=egl python video/human_motion_01/v2/create_previews.py
python video/human_motion_01/v2/finalize_mapping.py
```

这些步骤不训练策略。原视频、人体识别结果、第一版轨迹与已训练权重均保留。
"""
    (ROOT / "README.zh.md").write_text(
        f"<!-- tracks: README.md @ sha256:{sha(ROOT / 'README.md')[:16]} -->\n\n" + zh
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
