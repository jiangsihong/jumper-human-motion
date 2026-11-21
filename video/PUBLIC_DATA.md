# Public motion data

Raw source footage, clips and human/robot comparison media are user-supplied and
not redistributed. The source checksum in v1/manifest.json records the reviewed
input. The included derived landmarks and robot references describe only that
input and its manual adaptations. They are not a general video-to-policy model.

Robot-only previews, reference arrays, task material and selected checkpoints
are provided for inspection. To run extraction/comparison/final source checks,
supply the matching source video and regenerate clips. `v1/extract_clips.py`
accepts `--source`; both finalizers accept `--source` and require matching hashes.
No source verification is reported as passed when footage is missing.
