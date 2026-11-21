# Human motion 01: extracted video clips

Latest expressive mapping: [v2/README.md](../v2/README.md). This directory preserves v1; the parent launcher selects either version.

This document describes the original video-extraction stage. Human landmark
extraction, retargeting, references and previews are now documented in
[MAPPING.md](MAPPING.md). No reinforcement learning has been started.

The source is `../../b667ffa51c09b5d564d96c248cec54f4.mp4` (35.39 seconds reported
container duration, 1026 x 720, 30 fps). Its bytes are unchanged. See
`manifest.json` for the source SHA-256 and exact output frame intervals.

## Extracted performances

Times below refer to the original video. Intervals include their first frame
and exclude the end frame. Short neutral lead-ins and returns are retained;
talking-head introductions and the end card are excluded.

| Clip | Source interval | Duration | Frames |
| --- | --- | --- | --- |
| `clips/01_side_step.mp4` | 2.800--9.300 s | 6.500 s | 195 |
| `clips/02_kick_step.mp4` | 12.400--16.500 s | 4.100 s | 123 |
| `clips/03_arm_body_sway.mp4` | 24.900--32.300 s | 7.400 s | 222 |

1. Side step: repeated lateral evasive steps, arm movement and torso lean.
2. Kick step: crossing/kicking leg movement, balance changes and extended arms.
3. Arm/body sway: arm opening/lifting, torso sway and stepping.

The third performance contains edited close views around source time
28.37--30.30 s, which hide parts of the feet. These source edits remain in the
main clip. Do not treat it as an uninterrupted full-body motion-capture take.

An additional full-body-only excerpt is provided at
`clips/full_body/03_arm_body_sway_full_body.mp4`: source 24.900--28.333 s,
103 frames / 3.433 seconds. It ends before the first close-view cut and is a
partial excerpt, not the complete third performance or a verified motion loop.

## Processing and validation

- H.264 MP4, CRF 18, original 1026 x 720 resolution and 30 fps.
- Output clips are silent; the unchanged source retains its audio.
- No crop, mirror, speed change, interpolation, deblurring or watermark removal.
- Every output was fully decoded. Frame counts, frame rates and dimensions were
  checked; source and output SHA-256 values are recorded in `manifest.json`.
- `previews/` contains 12-frame contact sheets for manual review.

The cuts were selected through manual visual review, not automatic pose
recognition. These videos are source material for future motion mapping, not
training-ready reference trajectories.

## Reproduce

From the existing `pytorch` Conda environment:

```sh
python video/human_motion_01/v1/extract_clips.py
```

The script uses the installed `imageio`, `imageio_ffmpeg` and Pillow packages;
it writes the same clip files, previews and manifest again. It never launches
training or modifies the original source video.
