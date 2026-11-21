<!-- tracks: README.md @ sha256:9185db5dabf02cd5 -->

# 按动作名称查找模型权重

本目录集中存放选定的训练权重。每个动作有独立文件夹和同名 `.pt` 文件；原训练记录与中间权重保留在原来的 `logs/`。这些是原检查点的完整副本，只改存放位置和文件名，没有改写神经网络、优化器或训练状态。每个动作旁边的 `model_info.json` 记录来源、训练轮数和校验值。

| 文件 | 动作 | 任务名 |
| --- | --- | --- |
| `side_step/side_step.pt` | 闪身步 | `jumper.human_side_step` |
| `kick_step/kick_step.pt` | 浪子踢球 | `jumper.human_kick_step` |
| `arm_body_sway/arm_body_sway.pt` | 狗熊哆嗦毛 | `jumper.human_arm_body_sway` |
| `tripod_walk/tripod_walk.pt` | 三足步态行走 | `jumper.tripod` |

三个视频动作使用原来的 2000 轮权重，行走使用 2850 轮权重；这里只标记为训练候选，不宣称最好。狗熊哆嗦毛的前肢跟踪问题仍存在，先前提出的参考动作加有限修正方案尚未实施。改名整理不会改变训练效果。

## 按动作运行

在现有 pytorch 环境和工程根目录执行，一次运行一条：

```sh
conda activate pytorch
cd jumper-human-motion
python scripts/play.py --task jumper.human_side_step --checkpoint model/side_step/side_step.pt
python scripts/play.py --task jumper.human_kick_step --checkpoint model/kick_step/kick_step.pt
python scripts/play.py --task jumper.human_arm_body_sway --checkpoint model/arm_body_sway/arm_body_sway.pt
python scripts/play.py --task jumper.tripod --checkpoint model/tripod_walk/tripod_walk.pt
```

`--checkpoint` 要指向上面的具体文件，不能只写这个目录；现有的目录自动搜索规则查找的是 `model_*.pt`，明确指定文件则可以使用动作名称。

本次没有改启动入口或训练代码。后续训练仍将完整记录保存到 `logs/`，不会自动覆盖这里选定的副本。更换选定权重前，应先保留旧版本。
