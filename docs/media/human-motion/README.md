# Human-motion showcase and replay evidence

[English](README.md) | [简体中文](README.zh.md)

Recorded on 2026-11-21 from the four-file named-model layout. Only the three human
policies are shown here. No training, policy changes or best-take selection was performed.
The original training used Warp CUDA. This new evaluation uses native CPU because
CUDA was unavailable to the recording process. It does not validate backend parity.

## Files and meanings

- `source_full.mp4`: unchanged user-supplied input video.
- `*_source.mp4`: silent actual input excerpts. Sway uses only the 103 full-body frames.
- `*_source_mapping.mp4` / `.gif`: human versus v2 free-base PD reference; **not learned policy**.
- `*_policy.mp4` / `.gif`: one actual first attempt under the saved 2000-iteration policy.
- `*_joints.csv` / `.png`: measured joint state, velocity and torque from that same policy recording.
- `*_replay.json`: resolved replay settings; checkpoint paths made repository-relative.
- `manifest.json`: file/checkpoint SHA-256, commands, timings, backend and kinematic metrics.

The reference comparisons slow the human input 2x, 2x and 2.5x and add end holds.
Policy recordings run in simulation time at 30 fps; GIFs are 10 fps previews.
Independent GIFs do not start in sync. The MP4s have not been cut or speed-ramped.
GIF looping starts another viewing of the file, not another measured policy attempt.

## What the recordings establish

| Task | Checkpoint iterations | Control steps at boundary | MP4 frames at 30 fps | Video duration |
| --- | --- | --- | --- | --- |
| `jumper.human_side_step` | 2000 | 696 | 418 | 13.93 s |
| `jumper.human_kick_step` | 2000 | 456 | 274 | 9.13 s |
| `jumper.human_arm_body_sway` | 2000 | 474 | 284 | 9.47 s |

These first attempts reached the finite-clip boundary. `--stop-on-done` stops before
recording the wrapper's auto-reset pose; the terminal step is counted in metadata
but excluded from video/CSV. Sway still misses the large front-arm gesture. Reaching
the boundary does not establish similarity, robust balance or a multi-trial success rate.
The first two policies also retain tracking errors.

Kinematic metrics use joint CSVs with the first 0.5 s excluded. Joint names must
match the converted reference. Monitor t=0 is after one control step, so reference
frame k+1 matches CSV row k. Foot paths use forward kinematics of the same robot
with the floating base held at identity: these are body-relative limb spans, not
world travel or ground clearance. Kick replay RR X/Z spans are 119.1/47.4 mm.
Do not compare these directly with the older mapping report's differently defined
foot-site PD measurements. No external root assistance was added to the policy.

## Reproduce the recording

Activate your installed environment and run from the repository root. Observation
noise is on; the scene is the task default; physics/control are 200/50 Hz. One
unseeded, stochastic first attempt was recorded per task, so reruns may differ.

```sh
MUJOCO_GL=egl python scripts/play.py --task jumper.human_side_step --checkpoint model/side_step/side_step.pt --backend native --device cpu --cpu_threads 1 --num_envs 1 --headless --physics-hz 200 --steps 696 --stop-on-done --measure --video docs/media/human-motion/side_step_policy.mp4 --video-width 640 --video-height 480 --video-distance 0.85 --video-azimuth 135 --video-elevation -25 --replay-info-out docs/media/human-motion/side_step_replay.json
```

```sh
MUJOCO_GL=egl python scripts/play.py --task jumper.human_kick_step --checkpoint model/kick_step/kick_step.pt --backend native --device cpu --cpu_threads 1 --num_envs 1 --headless --physics-hz 200 --steps 456 --stop-on-done --measure --video docs/media/human-motion/kick_step_policy.mp4 --video-width 640 --video-height 480 --video-distance 0.85 --video-azimuth 135 --video-elevation -25 --replay-info-out docs/media/human-motion/kick_step_replay.json
```

```sh
MUJOCO_GL=egl python scripts/play.py --task jumper.human_arm_body_sway --checkpoint model/arm_body_sway/arm_body_sway.pt --backend native --device cpu --cpu_threads 1 --num_envs 1 --headless --physics-hz 200 --steps 474 --stop-on-done --measure --video docs/media/human-motion/arm_body_sway_policy.mp4 --video-width 640 --video-height 480 --video-distance 0.85 --video-azimuth 135 --video-elevation -25 --replay-info-out docs/media/human-motion/arm_body_sway_replay.json
```

Input footage retains its original visible attribution; it is not relicensed by the code license.
