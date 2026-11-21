<!-- tracks: README.md @ sha256:23d097856224134f -->

# Jumper 人体动作模仿实验

[English](README.md) | **简体中文**

本项目面向 22 自由度的 Jumper 螃蟹机器人，研究人体视频动作映射与 PPO 模仿训练。基于
[KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper) 二次开发，比较基线为上游提交
`198686eac0d188815d0943e9fee50de415edc80a`。

本项目的贡献是把人工审核的人体动作、机器人参考轨迹、训练任务和可追溯模型串成工程流程。
PPO、MuJoCo、MediaPipe、上游 mjlab/BeyondMimic 动作跟踪框架均为复用。

![第二版踢腿参考：PD 物理执行，不是训练策略成绩](video/human_motion_01/v2/previews/02_kick_step_pd_physics.contact_sheet.png)

## 相对原仓库的创新与工程扩展

| 方面 | 原仓库已有能力 | 本项目新增内容 |
| --- | --- | --- |
| 动作来源 | 机器人轨迹和示例任务 | 人工审核的视频片段、33 点人体关键点和视频动作参考生成 |
| 人体到机器人映射 | 机器人与参考轨迹基础设施 | 双臂对应 LF/RF，脚步改编为 LM/RM/LR/RR 四腿步序，并明确踢腿阶段 |
| 幅度检查 | 通用动作跟踪 | 并排保留保守版 v1、大幅度版 v2，分别记录计划姿态与实际 PD 物理执行 |
| 自定义任务 | 通用 PPO 和跟踪框架 | 三个独立的 human_* 任务，各自保存素材、观察输入和跟踪参数 |
| 地面一致性 | 四足共面检查 | 为迈步、踢腿增加可选近地支撑时序，标签和阈值参与缓存校验 |
| 有限动作结束 | 参考动作可循环并重置 | 在片尾自动跳回之前结束，避免把状态重置误认为连续完成动作 |
| 模型管理 | 按时间保存训练记录 | model/ 下按动作命名，记录来源、训练轮数和 SHA-256 |
| 可复现使用 | 原框架和示例 | 预览模型使用可移植路径，中英文说明及选定动作数据、权重随仓库提供 |

这些是工程流程和任务扩展，不是新发明 PPO，也不是任意视频端到端自动模仿。四足步序和单目踢腿深度包含人工适配。
原仓库说明保留为[英文版](docs/UPSTREAM_README.md)和[中文版](docs/UPSTREAM_README.zh.md)。

## 工作流程与公开内容

```text
用户自备视频 → 人工审核片段 → MediaPipe 人体关键点
  → 机器人目标轨迹 → 无额外机身支撑的 PD 物理记录
  → PPO 参考动作跟踪 → 测量回放 → 按动作命名的候选权重
```

- [v1](video/human_motion_01/v1/README.zh.md)：保守映射和最初的提取工具。
- [v2](video/human_motion_01/v2/README.zh.md)：大幅度迈步、踢腿参考。
- [model](model/README.zh.md)：按动作名称查找模型和来源。
- `tasks/jumper/human_*/media/`：训练真正读取的动作数据。

公开仓库不包含原始人物视频、裁剪、人体叠加及对照视频，也不包含下载的姿态依赖、日志和转换缓存；保留源文件校验记录。
重新提取或制作人与机器人对照时，需要提供有权使用的素材。现有机器人参考、机器人单独预览和策略回放不需要这些人物视频。
提取脚本中的固定时间段针对已审核素材，不适用于任意视频。

## 安装

开发时使用 Ubuntu x86_64、Python 3.12、RTX 5060 Laptop GPU（8 GiB）。GPU 训练需要兼容的 PyTorch CUDA 构建；
GPU/CPU 安装细节见[原项目环境指南](docs/AGENT_SETUP.md)。

```sh
git clone https://github.com/jiangsihong/jumper-human-motion.git
cd jumper-human-motion
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/train.py --list
```

已有 Conda 环境时，激活后执行可编辑安装即可。不要另外安装 PyPI 的 mjlab 或 rsl-rl-lib，本工程使用 rl/ 下的仓库内版本。

## 运行已保存的策略

在工程根目录一次运行一条：

```sh
python scripts/play.py --task jumper.human_side_step --checkpoint model/side_step/side_step.pt
python scripts/play.py --task jumper.human_kick_step --checkpoint model/kick_step/kick_step.pt
python scripts/play.py --task jumper.human_arm_body_sway --checkpoint model/arm_body_sway/arm_body_sway.pt
python scripts/play.py --task jumper.tripod --checkpoint model/tripod_walk/tripod_walk.pt
```

--task 选择任务及仿真配置，--checkpoint 指定训练网络。这里由策略实时控制物理仿真，不是播放保存的姿态。
三个视频动作是 2000 轮候选，行走是 2850 轮候选，用于检查，不宣称是最佳模型。

## 训练单个动作

```sh
python scripts/train.py --task jumper.human_side_step --backend warp --device cuda:0 --num_envs 256 --max-iterations 2000 --headless
```

把任务名改成 jumper.human_kick_step 或 jumper.human_arm_body_sway 可训练另外两段。建议依次训练。
2000 轮是初始预算，不保证学会。TensorBoard 由原入口启动，第一批数据写入后打开终端打印的网址。

完整记录按启动目录保存到 logs/jumper/任务名/时间戳/，所以建议从工程根目录运行。
后续训练不会自动替换 model/ 中的选定副本。--resume 继续最新记录，此时 --max-iterations 表示再训练多少轮。

## 查看映射参考

```sh
python video/human_motion_01/preview_mapping.py --version 1 --motion 2 --loop
python video/human_motion_01/preview_mapping.py --version 2 --motion 2 --loop
```

这里是记录好的参考回放，与已训练策略回放分开。第二版复用第一版的人体关键点和逆运动学工具。
第三段只用了最初 103 帧全身可见片段；第一、二段放慢为原速度的 1/2，第三段为 1/2.5。
重新裁剪前先提供原素材：

```sh
python video/human_motion_01/v1/extract_clips.py --source /path/to/source.mp4
```

人体提取还需要 MediaPipe、OpenCV 和 v1/.tools/ 下的官方 Pose Landmarker 模型，这些可选依赖不打包进仓库。
最终检查脚本要求源视频和片段的校验值匹配；素材缺失会明确报错，不会假称重新核验通过。

## 验证与当前限制

选定检查命令：

```sh
python -m pytest tests/test_registry.py tests/test_task_parity.py tests/test_reference_support.py tests/test_dance_motion.py -q
```

开发时这组检查记录为 51 项通过、1 项跳过。发布前连同翻译、目录布局和日志布局的检查共 139 项通过、1 项跳过；可移植预览模型及权重校验值检查也通过，Ruff 0.17.6 全仓静态检查通过。新增的三个支撑检查覆盖摆腿与支撑脚的几何区别、错误标签/时间/高度、缓存失效。
这些是软件检查，不是动作成功率试验或稳定性认证。

目前限制：

- 第三段训练候选仍未学好大幅摆臂。训练完成或总奖励上涨，不代表成功模仿；此前提出的“参考动作加有限修正”控制器尚未实施。
- 第二版参考 PD 回放的踢腿前后行程约 16.3 厘米、上下约 6.1 厘米，这不是已学习策略的成绩。
- 第三段参考约有 39 度机身倾斜。近地标签不证明真实承重接触数量或动态稳定性。
- 策略读取仿真位置误差和线速度；真机需要状态估计及部署接口，本项目没有真机验证结果。
- startup_validation.json 是早期五轮启动记录，与之后 model/ 中的 2000 轮候选区分。

## 来源与许可证

原 Jumper 框架、资产和声明保留 KingKong Robotics 及各贡献者署名。
见 [LICENSE](LICENSE)、[NOTICE](NOTICE)、[licenses](licenses/) 与[第三方代码来源](docs/VENDOR.md)。
保留原 Apache-2.0 和第三方许可条款；本仓库不会为用户自备视频授予 Apache 许可，也不重新分发这些素材。
