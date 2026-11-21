# Human-to-Jumper mapping, first version

Three video-derived motion mappings have been generated. No reinforcement
learning, policy optimization or new training-task registration was performed.
The original video, training code and existing checkpoints are unchanged.

## Inspect these files first

- `previews/*_comparison.mp4`: left = human video slowed 1.8x; right = robot
  free-base simple-PD physics playback. One-second lead-in and 1.5-second final
  transition/hold are included. These are not learned-policy demonstrations.
- `mapping/mapping_rules.json`: human-to-robot rules and numeric adaptations.
- `trajectories/<motion>/reference.npz`: primary reference in the existing dance
  loader's source format. Commands are mapped targets; `meas_*` are states
  actually reached in the bounded MuJoCo PD probe.
- `mapping/delivery_manifest.json`: final checks and reference hashes.

## Mapping decisions

MediaPipe Pose Landmarker 1.2.0 identified 33 landmarks per frame. Observations,
estimated world coordinates and visibility scores are in `poses/<motion>/`.
These monocular estimates are not ground-truth motion capture. The model and
isolated helper packages are in `.tools/`; the Conda training environment was
not modified. Model provenance is in each pose `metadata.json`.

| Human feature | Jumper mapping |
| --- | --- |
| Left shoulder/elbow/wrist | Left front arm `LF_J0..J3`, endpoint inverse kinematics |
| Right shoulder/elbow/wrist | Right front arm `RF_J0..J3`, endpoint inverse kinematics |
| Left hip/knee/ankle | Adapted stepping features for LF/LM/LR |
| Right hip/knee/ankle | Adapted stepping features for RF/RM/RR |
| Shoulder/hip centers and estimated torso orientation | Scaled free-base translation and roll/pitch/yaw, plus support-aware backward shift |
| Fingers | No human finger reconstruction; both gripper command targets remain at HOME=0 |

Anatomical landmark labels determine left and right; screen halves are not used
to guess them. The front pair are both arms and legs: arm gestures take priority,
and their walking contribution fades while they are raised. Middle/rear legs
carry an adapted gait, with a 0.22-second source-time delay for the rear pair.
This is not a one-to-one transfer of human two-legged locomotion. Rear/middle
foot lift is capped at 3.5 mm, lateral foot displacement at 9 mm, and mapped joint
speed at 4.5 rad/s. Large kicks become conservative foot/stance movements.

For the third motion, only source 24.900--28.333 seconds (103 frames, before the
close-view cut) is mapped. The full extracted third video is still preserved.
All durations are retimed by 1.8, then entry/exit and standing hold are added.

## Reference and playback checks

| Motion | Samples | Playback seconds | PD joint RMSE rad | Root XY RMSE mm |
| --- | --- | --- | --- | --- |
| 01_side_step | 708 | 14.16 | 0.0287 | 29.4 |
| 02_kick_step | 492 | 9.84 | 0.0233 | 3.7 |
| 03_arm_body_sway | 432 | 8.64 | 0.0324 | 8.3 |

All three bounded free-base PD probes had no fall under the recorded criterion
(base height below 55 mm or body-up cosine below 0.65). They use gravity and
resolved contacts, no root forces, kp=10 / kv=0.5 and a 1.7464 Nm torque limit.
They do not use the project's complete torque-speed servo model and do not prove
hardware execution, robust policy performance or full imitation accuracy.

`mapping/validation_report.json` audits the *planned* kinematic draft, including
lead-in/out, and records remaining target errors and geometry penetrations.
The largest planned pad/floor overlap is about 2.3 mm; no self-penetration was
observed in that prescribed full-reference audit. A near-ground target polygon
is only a design heuristic, not a stability certificate: many prescribed poses
have too few actual contacts. The subsequent physics probe resolves contacts
and supplies the measured reference channels. Root drift and tracking errors
remain; see `mapping/physics_probe.json` for the measured values.

Each primary reference passes `ensure_motion_npz(.02, directory,
reference="measured")`; converted MotionLoader data are in each `.cache/`.
The cached conversion's sample count matches the full measured trajectory.
This is data preparation, not a registered or trained new behavior.

## Files and reproduction

- `clips/`: trimmed source videos, unchanged by mapping.
- `poses/`: observed 33-point landmark arrays, visibility and feature CSVs.
- `mapping/*_robot_trajectory.npz`: planned qpos/qvel and requested/achieved IK foot points.
- `mapping/*_joints.csv`: human-readable planned robot joint angles in radians.
- `mapping/*_kinematic_source.npz`: schema-form draft before physical refinement.
- `mapping/*_pd_rollout.npz`: states reached in the simple-PD physics probe.
- `trajectories/`: one independent primary reference per motion.
- `previews/*_human_pose.mp4`: observed human landmark overlays.
- `previews/*_robot_reference.mp4`: prescribed kinematic target previews.
- `previews/*_pd_physics.mp4`: recorded physical playback, without learning.

Interactive replay (change `--motion` to 2 or 3):

```sh
conda activate pytorch
cd jumper-human-motion
python video/human_motion_01/v1/preview_mapping.py --motion 1 --loop
```

Default mode replays the recorded physical probe. `--mode planned` displays the
prescribed kinematic target instead; neither mode trains or executes a learned policy.

Rebuild mapping and playback, using the existing extracted poses:

```sh
cd jumper-human-motion
MUJOCO_GL=egl python video/human_motion_01/v1/build_motion_mapping.py --render
python video/human_motion_01/v1/check_mapping_physics.py
MUJOCO_GL=egl python video/human_motion_01/v1/create_mapping_previews.py
python video/human_motion_01/v1/finalize_mapping.py
```

The order matters: the mapping builder writes a kinematic draft; the physics
probe replaces `meas_*` with simulated observations. To extract poses again:

```sh
PYTHONPATH=video/human_motion_01/v1/.tools/pose_packages python video/human_motion_01/v1/extract_human_pose.py
```

Pose tool reference: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
