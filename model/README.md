# Named policy checkpoints

Use this directory to find the selected trained policy for each behavior.
Training histories and intermediate checkpoints remain in their original `logs/`
directories. Each named file is a byte-identical full-checkpoint copy: the policy,
optimizer and training state have not been rewritten. `model_info.json` records
the source path, iteration and SHA-256 for each one.

| Directory | Behavior | Task |
| --- | --- | --- |
| `side_step/side_step.pt` | Side step | `jumper.human_side_step` |
| `kick_step/kick_step.pt` | Kick step | `jumper.human_kick_step` |
| `arm_body_sway/arm_body_sway.pt` | Arm/body sway | `jumper.human_arm_body_sway` |
| `tripod_walk/tripod_walk.pt` | Tripod walking | `jumper.tripod` |

The three human clips are the original 2000-iteration candidates; tripod walking
is the 2850-iteration checkpoint. They are not labeled best. The existing sway
policy's front-arm tracking problem is unchanged; its proposed residual-control
solution has not been implemented. Copying/renaming does not improve a policy.

## Replay by action name

From the existing pytorch environment and project root:

```sh
conda activate pytorch
cd jumper-human-motion
python scripts/play.py --task jumper.human_side_step --checkpoint model/side_step/side_step.pt
python scripts/play.py --task jumper.human_kick_step --checkpoint model/kick_step/kick_step.pt
python scripts/play.py --task jumper.human_arm_body_sway --checkpoint model/arm_body_sway/arm_body_sway.pt
python scripts/play.py --task jumper.tripod --checkpoint model/tripod_walk/tripod_walk.pt
```

Run one command at a time. Supply the named **file**, not just its directory:
the existing directory search looks for `model_*.pt`, while an explicit file can
have any name. No launcher or training code was changed. Future training still
writes its complete run under `logs/`; it does not automatically replace these
selected copies. Preserve an old version before selecting a replacement.
