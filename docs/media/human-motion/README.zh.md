<!-- tracks: README.md @ sha256:5529d6f5de0d8704 -->

# 人体动作展示与策略回放证据

[English](README.md) | 简体中文

2026-11-21 从已命名的模型目录录制，仅展示三个视频动作策略。本次没有训练、修改策略或挑选最佳尝试。
原训练使用 Warp CUDA；新录像进程无法使用 CUDA，所以使用 native CPU 回放，不代表已验证两后端一致。

## 文件含义

- `source_full.mp4`：未修改的用户提供原视频。
- `*_source.mp4`：无声的实际输入片段；第三段只取全身可见的 103 帧。
- `*_source_mapping.mp4` / `.gif`：人物与 v2 无额外机身支撑的 PD 参考对照，**并非训练策略**。
- `*_policy.mp4` / `.gif`：2000 轮权重控制下的一次真实策略尝试。
- `*_joints.csv` / `.png`：同一次录像的关节位置、速度及力矩。
- `*_replay.json`：实际回放设置；权重路径改为仓库相对路径。
- `manifest.json`：文件/权重校验值、命令、时间、后端及运动学测量。

人物/参考对照将人物放慢 2、2、2.5 倍并加入首尾停留。策略录像按仿真时间录为 30 帧/秒；动图为 10 帧/秒预览。
独立动图并不同步。MP4 没有剪掉中间动作或变速；动图循环是重复观看文件，不是新的策略实验。

## 本次录像可以说明什么

| 任务 | 权重训练轮数 | 片尾控制步数 | MP4 帧数（30 帧/秒） | 视频时长 |
| --- | --- | --- | --- | --- |
| `jumper.human_side_step` | 2000 | 696 | 418 | 13.93 秒 |
| `jumper.human_kick_step` | 2000 | 456 | 274 | 9.13 秒 |
| `jumper.human_arm_body_sway` | 2000 | 474 | 284 | 9.47 秒 |

这三次尝试都到达有限参考的片尾。`--stop-on-done` 在记录自动重置姿态前停止；终止步计入元数据，
但不会写进视频和 CSV。第三段仍缺少大幅前肢摆动。到达片尾不等于动作相似、稳定性充分或多次成功率。
前两个动作也存在跟踪偏差。

运动学测量排除 CSV 前 0.5 秒，关节名必须与转换参考一致。记录器 t=0 是首个控制步之后，
所以 CSV 第 k 行对应参考第 k+1 帧。把浮动基座固定为单位姿态，对同一模型作关节正运动学，得到机身相对足端行程，
不是世界位移或离地高度。踢腿策略的右后腿 X/Z 行程为 119.1/47.4 毫米。
此口径不能直接与旧映射报告的足部 site PD 测量比较。本次没有给策略添加额外机身支撑。

## 复现录像

激活已安装依赖的环境，在仓库根目录执行。观察噪声开启，场景为任务默认，物理/控制为 200/50 Hz。
每个任务只记录一次未固定随机种子的尝试，存在随机噪声，重跑可能不同。

```sh
MUJOCO_GL=egl python scripts/play.py --task jumper.human_side_step --checkpoint model/side_step/side_step.pt --backend native --device cpu --cpu_threads 1 --num_envs 1 --headless --physics-hz 200 --steps 696 --stop-on-done --measure --video docs/media/human-motion/side_step_policy.mp4 --video-width 640 --video-height 480 --video-distance 0.85 --video-azimuth 135 --video-elevation -25 --replay-info-out docs/media/human-motion/side_step_replay.json
```

```sh
MUJOCO_GL=egl python scripts/play.py --task jumper.human_kick_step --checkpoint model/kick_step/kick_step.pt --backend native --device cpu --cpu_threads 1 --num_envs 1 --headless --physics-hz 200 --steps 456 --stop-on-done --measure --video docs/media/human-motion/kick_step_policy.mp4 --video-width 640 --video-height 480 --video-distance 0.85 --video-azimuth 135 --video-elevation -25 --replay-info-out docs/media/human-motion/kick_step_replay.json
```

```sh
MUJOCO_GL=egl python scripts/play.py --task jumper.human_arm_body_sway --checkpoint model/arm_body_sway/arm_body_sway.pt --backend native --device cpu --cpu_threads 1 --num_envs 1 --headless --physics-hz 200 --steps 474 --stop-on-done --measure --video docs/media/human-motion/arm_body_sway_policy.mp4 --video-width 640 --video-height 480 --video-distance 0.85 --video-azimuth 135 --video-elevation -25 --replay-info-out docs/media/human-motion/arm_body_sway_replay.json
```

人物素材保留原画面署名，代码许可证不为该视频重新授权。
