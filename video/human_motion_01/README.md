# Human motion 01: version navigation

The first mapping is in `v1/`; the expressive second mapping is in `v2/`.
Both versions retain their motion data, reports and preview videos. This was a
file-layout migration: motion arrays, media, original source and training copies
were not regenerated. Embedded NPZ provenance remains the historical record of
where files were originally generated; current JSON report paths were relocated.

| Directory | Contents |
| --- | --- |
| [v1/](v1/README.md) | First conservative mapping, original extraction/pose data, scripts and tools |
| [v2/](v2/README.md) | Second expressive mapping, trajectories and comparisons |

The second version reuses `v1/poses/`, `v1/clips/`, the v1 IK utility and preview
model. These are explicit sibling-directory references; retain both versions.
The original source video remains one directory above this navigation page.

## Preview either version

```sh
conda activate pytorch
python video/human_motion_01/preview_mapping.py --version 1 --motion 2 --loop
python video/human_motion_01/preview_mapping.py --version 2 --motion 2 --loop
```

Run one command at a time. Omitting `--version` selects v2. Each version also has
its own `preview_mapping.py`. These commands play mapped reference records.
Policy training/replay still uses the separate `tasks/jumper/human_*` tasks and
their own `media/` data. The task inputs and checkpoint locations are unchanged.

Public distribution excludes raw human footage; see [data notes](../PUBLIC_DATA.md).
