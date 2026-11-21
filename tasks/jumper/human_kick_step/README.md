# Human performance: 02_kick_step

This independent simulation-only task imitates clip 02_kick_step. The media copy is
byte-identical to the v2 measured physics reference; no amplitude was reduced.
The two front arms follow the human arms and the other four legs the adapted
human footwork. PPO learns reference tracking, not video pixels.

## Start from the existing environment

```sh
conda activate pytorch
cd jumper-human-motion
python scripts/train.py --task jumper.human_kick_step --backend warp --device cuda:0 --num_envs 256 --max-iterations 2000 --headless
```

Run the three tasks sequentially. 2000 is an initial budget, not guaranteed
convergence or a runtime estimate. Open the TensorBoard URL printed by training.
Starting in the repository root keeps the logs under
`logs/jumper/jumper.human_kick_step/<timestamp>/`.

## Replay the newest policy after training

```sh
python scripts/play.py --task jumper.human_kick_step --backend warp --device cuda:0 --num_envs 1 --measure --steps 456 --stop-on-done
```

This uses policy-controlled physics, not recorded pose playback. Measure a single
attempt from the first frame before the command wraps; the task also ends before
wraparound. Check limb amplitude, foot lift, tracking error and full completion.
A random four-second training window completing is not a whole-clip success.

To continue a stopped run, add `--resume` to the training command. In that case
`--max-iterations` is the number of additional iterations. Do not use the old
walking checkpoint as this task's state and action observations differ.

## Reference validation and limitations

The original fixed-four-foot coplanarity check correctly rejects these stepping
references. This task supplies an explicit schedule of near-ground feet instead:
each frame has at least two labeled feet within 5 mm of standing foot-site height,
and no foot site lies more than 3.5 mm below that height. Schedule content, rate
and tolerances are part of the conversion cache fingerprint; default tasks keep
the original check. Joint limits, continuity and model consistency checks remain.
The labels are geometric proxies, not contact forces or stability certification.
The right-rear leg retains roughly 163 mm local fore/aft and 61 mm vertical
range. A dedicated right-rear tracking term prevents averaging its motion away.

The actor reads simulated position error and linear velocity. Real deployment
requires a corresponding estimator and a supported controller contract; this
task is not hardware-ready. There are no operator command axes or controls.yaml.

## Startup validation

On RTX 5060 Laptop GPU (8 GiB), warp:cuda, 256 environments, five iterations
completed. Mean reward: 0.18 -> 0.15. This only proves startup; a learned
performance is not claimed. See `startup_validation.json` for the actual short-run
checkpoint and replay measurements. No long formal training was left running.

Shared-loader, registry and parity validation: 51 passed, 1 skipped. Independent
read-only review found no startup blocker. Existing side-step and dance tasks,
source videos and model weights are preserved.

The noisy startup-policy replay ended early at step 186 of 456
(3.72 simulated seconds), with 185 measurement rows. It has not
learned the performance; the saved checkpoint is for startup verification only.
