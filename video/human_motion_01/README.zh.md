<!-- tracks: README.md @ sha256:9b2671c38de132d9 -->

# 人体动作 01：版本入口

第一版映射在 `v1/`，第二版大幅度映射在 `v2/`，两者为平级目录。两版的轨迹、检查报告和预览视频均保留。本次只调整文件布局，没有重新生成动作数组、视频、原始素材或训练任务中的副本。NPZ 内嵌的来源信息保留生成时的历史路径；JSON 报告中用于访问文件的路径已经同步更新。

| 目录 | 内容 |
| --- | --- |
| [第一版 v1](v1/README.zh.md) | 原保守映射，以及最初的视频裁剪、人体关键点、脚本和工具 |
| [第二版 v2](v2/README.zh.md) | 大幅度映射、轨迹和对照视频 |

第二版仍复用 `v1/poses/`、`v1/clips/`、第一版逆运动学工具和预览模型，引用已改为明确的平级目录路径。请同时保留两版。原始视频仍在本入口目录的上一级。

## 选择版本预览

```sh
conda activate pytorch
python video/human_motion_01/preview_mapping.py --version 1 --motion 2 --loop
python video/human_motion_01/preview_mapping.py --version 2 --motion 2 --loop
```

一次执行一条。省略 `--version` 时默认打开第二版。两版目录中也各有独立的 `preview_mapping.py`。这些命令播放映射参考记录；策略训练、回放仍使用单独的 `tasks/jumper/human_*` 任务及其 `media/` 数据，训练输入和已有权重位置不变。

公开仓库不包含原人物视频，见[数据说明](../PUBLIC_DATA.zh.md)。
