<!-- tracks: README.md @ sha256:2bb91cd29f0ea5e7 -->

# 人体闪身步模仿训练

这个任务训练你选中的第一段闪身步，不是项目自带的示范舞蹈。两条前肢对应人的双臂，其余四条腿对应改编后的脚步。训练素材 `media/side_step.npz` 与第二版第一段实际物理记录完全一致，机身左右行程约 110.3 毫米，没有为了训练而缩小动作。第二、三段尚未接入这个任务。

方法是 PPO 动作跟踪：根据当前身体状态、参考关节和未来目标学习控制电机，不是直接学习视频像素。策略还读取仿真中的位置误差和线速度；真机需要对应的状态估计器。因此当前是仿真训练任务，不能直接认为已支持真机。

## 启动一次新训练

继续使用现有环境，不需要新建虚拟环境或安装训练依赖。在 zsh 终端运行：

```sh
conda activate pytorch
cd jumper-human-motion
python scripts/train.py --task jumper.human_side_step --backend warp --device cuda:0 --num_envs 256 --max-iterations 2000 --headless
```

命令会创建新训练记录。TensorBoard 会由现有入口启动，打开终端打印的网址即可。这里使用 256 个环境，每轮 PPO 采集 8192 个样本，不等同于默认 4096 环境的配置。2000 轮是初始训练预算，不保证届时学会，也不是耗时估算。除了总奖励，还要看关节和身体跟踪误差、动作幅度与完整完成情况。

按 `Ctrl+C` 停止。每 50 轮和正常训练结束时保存权重。日志目录为 `logs/jumper/jumper.human_side_step/时间戳/`。

## 回放最新权重

```sh
conda activate pytorch
cd jumper-human-motion
python scripts/play.py --task jumper.human_side_step --backend warp --device cuda:0 --num_envs 1 --measure --steps 696 --stop-on-done
```

这次是策略实时控制电机并进行物理仿真，不是播放已经录好的映射姿态。观察噪声保持开启。原参考有 698 帧、50 Hz；只检查第一次完整尝试，在片尾自动重置前结束，任务自身也加入了片尾终止。要检查完整完成率、机身左右行程、四足抬脚和姿态误差，原地站着或提前结束不能算学会。训练时会随机采样六秒片段，因此训练中完成一个片段不等于从第一帧完成整段。

## 接着训练

```sh
python scripts/train.py --task jumper.human_side_step --backend warp --device cuda:0 --num_envs 256 --resume --max-iterations 1000 --headless
```

`--resume` 继续最新记录；这里的 1000 是再训练 1000 轮。若指定某个权重，用 `--checkpoint /绝对路径/model_N.pt` 替代 `--resume`。不要接着旧行走任务的权重训练，本任务的观察输入不同。

## 实际检查结果

RTX 5060 Laptop GPU（8 GiB）、warp:cuda、256 环境的五轮短训练已跑通，总奖励 .24 → .81。这只是启动检查，不代表学会。五轮策略在开启观察噪声、单环境回放中于第 656 步提前结束，未完成要求的 696 步；仿真时间 13.12 秒，保存了 655 行测量。具体权重和测量路径见 `startup_validation.json`。

GPU、MuJoCo、依赖位置、任务注册四项训练检查通过；注册和步态分类检查共 25 项通过。素材与第二版原参考校验值一致，独立只读复核未发现启动阻断。没有留下长时间正式训练在后台运行；也没有安装或验证本任务的真机部署工具链。

## 文件作用

- `__init__.py`：注册任务。
- `env_cfg.py`：素材位置、奖励、观察与扰动设置。
- `rl_cfg.py`：PPO 参数和动作跟踪训练器。
- `mdp/__init__.py`：在片尾自动跳回之前结束。
- `media/side_step.npz`：本次素材；`.cache/` 是自动生成的转换缓存。
- `startup_validation.json`：启动验证结果，不代表已训练成功。

任务由固定参考动作驱动，没有操作员速度指令，所以无需 `controls.yaml`。现有步态任务、已有权重、原视频和另外两段映射保留。
