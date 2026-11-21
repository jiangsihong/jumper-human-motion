<!-- tracks: PUBLIC_DATA.md @ sha256:403394d14b0ff3f9 -->

# 公开动作数据

按项目作者要求，`docs/media/human-motion/` 展示原始输入视频、三个实际使用的片段、人物/PD 对照、
训练后的策略回放、动图和关节测量。原始视频保持不变；提取片段无声，保留画面原有署名/水印。
人物视频是演示输入，不能当成机器人策略输出。其版权归原权利人；代码的 Apache-2.0 许可不为人物视频重新授权。

`human_motion_01/v1/manifest.json` 保存审核输入的校验值。`docs/media/human-motion/manifest.json`
记录展示文件、权重校验值、录像命令及测量。输入、PD 参考、已训练策略分别标明。
第三段实际输入只取全身可见的片段。

下载的姿态依赖、转换缓存和完整训练日志仍不打包。重新生成提取工作区时，把本仓库
`docs/media/human-motion/source_full.mp4` 传给 `v1/extract_clips.py --source`。
最终检查要求源视频和片段校验值匹配；现有机器人参考回放不需要重新提取人体姿态。
