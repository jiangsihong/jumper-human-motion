# Public motion data

At the project author's request, the showcase under `docs/media/human-motion/`
includes the original input video, three actual input excerpts, human/PD comparisons,
trained-policy rollouts, GIFs and joint measurements. The original video remains
unaltered; excerpts are silent and preserve visible source attribution/watermarks.
The footage is demonstration input, not original robot-policy output. Copyright
remains with its original rightsholders; Apache-2.0 for the code does not relicense it.

The source checksum in `human_motion_01/v1/manifest.json` records the reviewed
input. `docs/media/human-motion/manifest.json` links showcase files, checkpoint
hashes, recording commands and measurements. Source, PD reference and trained
policy are labeled separately. Clip 3's actual input is only its full-body excerpt.

Downloaded pose packages, conversion caches and full training logs remain excluded.
To recreate the extraction workspace, run `v1/extract_clips.py --source` with the
included `docs/media/human-motion/source_full.mp4`. Finalizers require matching
source/clip hashes. Existing robot-reference playback does not need pose extraction.
