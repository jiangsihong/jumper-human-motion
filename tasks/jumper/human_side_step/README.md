# Human side-step imitation

Train the first user-selected v2 clip, not the repository's bundled demo dance.
The two front arms follow human arms and the other four legs follow the adapted
human footwork. `media/side_step.npz` is an exact copy of the measured simulation
reference in `video/human_motion_01/v2/trajectories/01_side_step/reference.npz`.
Its recorded body Y range is 110.3 mm; the motion has not been shrunk for training.
This task covers clip 1 only. Clips 2 and 3 are not registered for training here.

The method is PPO reference tracking using the existing motion-tracking runner,
not supervised cloning of video pixels. The actor observes current proprioception,
reference joints/future targets, root position error and simulated linear velocity.
The last two require a suitable state estimator before hardware deployment. This
is a simulation-only task and must not be reported as hardware-ready.

## Start a fresh training run

Use the existing environment; no additional environment or dependencies required.
From a zsh terminal:

```sh
conda activate pytorch
cd jumper-human-motion
python scripts/train.py --task jumper.human_side_step --backend warp --device cuda:0 --num_envs 256 --max-iterations 2000 --headless
```

This starts a fresh run. TensorBoard starts through the existing entry point; use
the URL it prints. Initial settings target 256 environments (8192 samples per PPO
iteration), not the default 4096-environment recipe. 2000 iterations is an initial
budget, not a guaranteed convergence point or a time estimate. Check joint/body
tracking errors, amplitude and full-clip completion, not total reward alone.

`Ctrl+C` stops training. Checkpoints are saved every 50 iterations and at normal
completion. Logs go to `logs/jumper/jumper.human_side_step/<timestamp>/`.

## Replay the newest checkpoint

```sh
conda activate pytorch
cd jumper-human-motion
python scripts/play.py --task jumper.human_side_step --backend warp --device cuda:0 --num_envs 1 --measure --steps 696 --stop-on-done
```

This is actual policy control with physics, not recorded mapping playback.
Observation noise remains enabled. The reference has 698 frames at 50 Hz;
evaluate a single attempt before the command can wrap and reset the state.
The task also terminates before wraparound. Check full completion, body Y range,
four-foot lift and pose error; a stationary or early-ending attempt is not success.
Training uses sampled six-second windows, so completing a random training window
is not equivalent to completing the full performance from its first frame.

## Continue a stopped run

```sh
python scripts/train.py --task jumper.human_side_step --backend warp --device cuda:0 --num_envs 256 --resume --max-iterations 1000 --headless
```

`--resume` selects the newest run; `--max-iterations 1000` adds 1000 iterations.
For a particular run use `--checkpoint /absolute/path/to/model_N.pt` instead of
`--resume`. Do not resume the old walking task: this task has different inputs.

## What has actually been checked

On RTX 5060 Laptop GPU (8 GiB), warp:cuda, 256 environments: a five-iteration smoke
run completed. Mean reward was .24 -> .81. This is startup evidence only: the
five-iteration policy has not learned the choreography. Its noisy one-environment
replay ended at step 656 of the requested 696 (13.12 simulated seconds), with 655
measurement rows. No full-performance success is claimed. See
`startup_validation.json` for the exact checkpoint and measurement paths.

The GPU/MuJoCo/import/registry training gates passed; registry and task-parity
checks passed (25 tests). Media hashes match the selected v2 reference exactly.
An independent read-only review found no startup blocker. No long formal run was
left running. Deployment toolchains were not installed or validated for this task.

## Files

- `__init__.py`: lightweight registration.
- `env_cfg.py`: reference location, rewards, observations and robustness choices.
- `rl_cfg.py`: PPO settings and tracking runner.
- `mdp/__init__.py`: finish before the reference wraps and teleports.
- `media/side_step.npz`: selected motion; `.cache/` is an automatic conversion.
- `startup_validation.json`: measured startup result, not a trained-policy claim.

There are no operator command axes, so no `controls.yaml` is needed for this
finite reference-driven task. Existing gait tasks, their weights, source videos
and the other two mapped clips are unchanged.
