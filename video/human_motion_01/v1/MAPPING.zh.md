<!-- tracks: MAPPING.md @ sha256:302017270226f3ea -->

# 人体到 Jumper 的映射说明（第一版）

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
| 01_side_step | 708 | 14.16 | 0.0287 | 29.4 |
| 02_kick_step | 492 | 9.84 | 0.0233 | 3.7 |
| 03_arm_body_sway | 432 | 8.64 | 0.0324 | 8.3 |

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
