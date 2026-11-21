# Expressive mapping v2

This revision uses human arm observations only for LF/RF and human steps only
for LM/LR and RM/RR. It preserves much larger foot trajectories and the reviewed
right-leg retract/forward-swing kick. No learning or training-task registration
was performed. Version 1 is preserved in the sibling `../v1/` directory.

## View the result

- `previews/*_comparison.mp4`: human versus prescribed mapping target.
- `previews/*_physics_comparison.mp4`: human versus actual free-base PD execution.
- Labels distinguish target animation from physical execution. No root support
  force, learned policy, or hidden assistance is used in physical execution.
- Camera 115 degrees shows the right-rear kick and the front arms clearly.

## Mapping and amplitude

| Human part | Robot part |
| --- | --- |
| Left arm | LF only |
| Right arm | RF only |
| Left foot/step | LM and LR |
| Right foot/step | RM and RR; RR performs the principal kick |
| Torso and balance | Shared root pose; planted-foot compensation |

The four legs use a wider 210 mm lateral stance. Walking clearance targets are
45 mm for the side steps and 55 mm for the body-sway steps. RR's kick retracts
to x=-200 mm then swings forward to x=-40 mm at z=70 mm, versus the 10 mm
standing foot-site height. The 160 mm fore/aft stroke is preserved. A 20 mm
outward displacement makes that height reachable without the old knee-stop
singularity. Front-limb targets never receive foot-step features.

The second clip mainly kicks with the human right leg. LM/LR therefore support
that performance rather than inventing simultaneous left-leg kicks. RM performs
the preparation and RR the large kick. Motions 1 and 3 visibly step with all four
legs. Shared body and balance changes can move all joints without changing this
semantic separation.

Source landmarks are reused from `../v1/poses/`. This is a video-feature
mapping with manually authored crawl contact order and reviewed kick phases,
not a frame-exact reconstruction of human two-legged motion. Clip 3 still uses
the first 103 full-body frames. Timing is slowed 2x for clips 1/2 and 2.5x for
clip 3; half-second initial/final holds are added. Clip 3's final steps land
before its final large arm gesture, preserving the lift while improving balance.

## Measured physical motion

| Clip | Body Y range mm | RR X range relative to body mm | RR Z range relative to body mm | Joint RMSE rad |
| --- | --- | --- | --- | --- |
| 01_side_step | 110.3 | 35.0 | 64.4 | 0.0396 |
| 02_kick_step | 20.4 | 162.8 | 61.0 | 0.0315 |
| 03_arm_body_sway | 37.1 | 60.1 | 113.6 | 0.0737 |

All three single bounded probes did not cross the fall criterion (base below
55 mm or body-up cosine below .65). The simulator uses gravity/contact and simple
kp10/kv.5 PD actuators with 1.7464 Nm torque limits. It is not the full servo
torque-speed model. Clip 3 reaches roughly 39 degrees of body tilt and retains
more tracking error; do not describe these runs as comprehensive stability tests.

The planned IK targets are intentionally expressive, and not all targets are
reachable exactly: the largest original target residuals are about 58, 44 and
105 mm. Planned pose geometry includes overlaps up to about 4.1 mm in clip 1.
These are recorded in `mapping/validation_report.json`, not silently marked as
passed. The actual PD records resolve contacts and differ from those targets;
both versions are retained so visual similarity and execution can be judged.

## Reference data

- `mapping/mapping_rules.json`: semantic rules and numerical settings.
- `mapping/*_robot_trajectory.npz`: achieved planned qpos, before dynamic execution.
- `mapping/*_pd_rollout.npz`: actual free-base PD execution states and foot paths.
- `mapping/*_planned_reference.npz`: source-format planned copy.
- `trajectories/<clip>/reference.npz`: planned command channels plus physically
  observed `meas_*` channels. The measured tracks preserve the large motion;
  they are not the old small-motion fallback. They are simulation measurements,
  not hardware measurements or learned-policy trajectories.
- `mapping/physics_report.json`: physical amplitude, body motion and fall results.
- `mapping/delivery_manifest.json`: amplitude gates, reference hashes and status.

Source-format loading passes for all three clips. The existing dance converter
accepts clip 1, but rejects clips 2 and 3 at its four-support-foot coplanarity
check. Passing that check alone does not validate stepping or kicking behavior.
These motions need a future task with scheduled contacts before training; do
not shrink them to pass an incompatible fixed-support check. No training
command is supplied here.

## Interactive viewing

```sh
conda activate pytorch
python video/human_motion_01/v2/preview_mapping.py --motion 2 --loop
```

Default is recorded physics; `--mode planned` views the prescribed target.
Change `--motion` to 1 or 3 for the other clips. Neither mode trains a policy.

## Reproduce

From `/home/jm/code/jumper` in the existing pytorch environment:

```sh
python video/human_motion_01/v2/build_mapping.py
python video/human_motion_01/v2/check_physics.py
MUJOCO_GL=egl python video/human_motion_01/v2/create_previews.py
python video/human_motion_01/v2/finalize_mapping.py
```

The builder writes expressive kinematic candidates; the finalizer retains them
and writes the physically observed measured channels without changing the motion
targets. Source videos, pose estimates, v1 data and existing model checkpoints
remain unchanged.
