# hdf52lerobot

Convert the raw per-stream HDF5/MKV recordings written by the bimanual handheld
UMI rig into a [LeRobot](https://github.com/huggingface/lerobot) **v3.0** dataset,
structurally identical to the `TacCap` datasets (20-dim `observation.state`/
`action`, videos aggregated per camera key) but tagged `robot_type:
bi_rdt_gripper`, since the rig's gripper and planar tactile sensors are
different hardware.

No `lerobot` install is needed: the v3.0 layout is written directly, and PyAV
does the transcoding.

## Installation

```bash
git clone https://github.com/XenseRobotics-AI/xense-dataset-convert.git
cd xense-dataset-convert
pip install -e .

hdf52lerobot --help
```

## Usage

```bash
hdf52lerobot \
  --input /data/SuHang \
  --output /data/SuHang_lerobot \
  --task "your task description"
```

`--input` accepts either one episode directory or a parent holding several
(sorted by name, which becomes the episode order); repeat the flag for more.
Two 72-second episodes (11 streams each, 1.1 GB of source) convert in ~14 s and
land at ~118 MB.

Useful flags: `--config` (rule file, default
[`configs/hdf52lerobot.yaml`](../configs/hdf52lerobot.yaml)), `--fps`,
`--quat-order`, `--pair`, `--max-sync-gap-frames`, `--warn-sync-gap-frames`,
`--crf`, `--gop`, `--preset`, `--workers`, `--overwrite`.

## What it reads

| Source file | Becomes |
|---|---|
| `head_*_mvs_mono_*.mkv` | `observation.images.head` |
| `{side}_umi_*_mvs_mono_*.mkv` | `observation.images.{side}_wrist` |
| `{side}_umi_*_tactile_{left,right}_*.mkv` | `observation.images.{side}_tactile_{left,right}` |
| `{side}_umi_*_tracker_*.hdf5` | `{side}_tcp.x/y/z` + `.r1..r6` in `observation.state` |
| `{side}_umi_*_openv2_{lo}_{hi}.hdf5` | `{side}_gripper.pos` in `observation.state` |
| `*.mkv.timestamp.hdf5` | per-frame timestamps used for synchronization |

## What it writes

```
<output>/
├── meta/
│   ├── info.json                          # fps, robot_type, features
│   ├── stats.json                         # min/max/mean/std/count per feature
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet  # per-episode ranges + stats/* blocks
├── data/chunk-000/file-000.parquet        # episodes aggregated, rolls at 100 MB
└── videos/observation.images.<key>/chunk-000/file-000.mp4   # rolls at 200 MB
```

## Conversion conventions

- **Rotation**: the tracker file's 7 columns are `[x, y, z, qx, qy, qz, qw]`, so
  the quaternion is read as `xyzw` (`--quat-order wxyz` for a rig that stores the
  scalar first). It becomes the 6D representation LeRobot's `bi_taccap_gripper`
  uses — `r1..r3` is the first column of the rotation matrix, `r4..r6` the second.
- **Gripper**: `clip((v - lo) / (hi - lo), 0, 1)` with the calibration bounds read
  from the file name; `hi` is fully open, so `1.0` = open, matching TacCap.
- **Action**: `action[t] = state[t+1]`, with the final frame repeating its own state.
- **Time axis**: `timestamp = frame_index / fps`, and `fps` is declared (default 20),
  never inferred — the rig's per-device arrival timestamps jitter too much for a
  median estimate to be stable.
- **Pairing**: `--pair index` (default) pairs frame *i* of every stream with frame
  *i* of the master clock, which is right when all streams share one trigger and
  have equal length. `--pair nearest` re-pairs by real timestamps instead, and is
  used automatically for any stream whose length differs from the master's.
  `--pair grid` is for recorders whose loop does not run at the declared fps: it
  pairs by iteration like `index`, then lays a uniform grid at `fps` over the
  episode and gives every tick the nearest iteration. A tick inside a stall
  repeats the previous iteration, iterations between ticks are dropped, and
  both counts are logged per stream and written to `meta/sources.json`.
  Either way the pairing is validated against the real timestamps: beyond
  `--warn-sync-gap-frames` (1.5) frames get reported, beyond
  `--max-sync-gap-frames` (3.0) the episode is rejected.
- **Source time**: with `keep_source_timestamp: true` every frame also carries
  `observation.source_timestamp`, the recorder's absolute time (master clock,
  int64 nanoseconds since the Unix epoch) of the iteration it came from.
  `timestamp` stays `frame_index / fps` as LeRobot requires; this column keeps
  the real time axis. It is an int64 rather than float64 seconds because
  LeRobot's torch transform hands float scalars back as float32, which cannot
  hold a Unix time to better than a few seconds.
- **Provenance**: `meta/sources.json` records the tool version, the full config,
  the inputs and, per episode, the recording directory, its files, frame counts,
  duplicated / dropped counts and tick offsets. LeRobot ignores extra files
  under `meta/`.
- **Video**: all streams are re-encoded (not remuxed) to H.264 / yuv420p / CFR at
  the declared fps with a short GOP — the sources carry 200+ frame GOPs, which
  makes LeRobot's random-access reads crawl, and LeRobot's default
  `tolerance_s = 1e-4` requires frame *i* to sit exactly at *i/fps*.

## Caveats

- The tactile sensors are recorded losslessly (FFV1) and are re-encoded to H.264
  at `--crf 23` (~5× smaller, mean error ~1.7/255). That matches TacCap, which
  also stores tactile as H.264 — lower `--crf` if your downstream model is
  sensitive to it.
- The time axis is index-based, so a recording that stalls mid-episode is written
  as if it ran at constant speed. The converter logs the measured warp against a
  uniform grid so you can see how large it is per episode.

## Profiles

[`configs/tacverse_rdt.yaml`](../configs/tacverse_rdt.yaml) is the profile for
the TacVerse-RDT recordings: same rig layout, `pair: grid` at 20 Hz because that
recorder's loop stalls, and `keep_source_timestamp: true`.

```bash
hdf52lerobot --config configs/tacverse_rdt.yaml \
  --input /archive/TacVerse-RDT/035 --input /archive/TacVerse-RDT/036 \
  --output /archive/TacVerse/<name> --task "<task text>"
```

## Extending it (new cameras, head pose)

[`configs/hdf52lerobot.yaml`](../configs/hdf52lerobot.yaml) holds the filename rules
and the `state_layout`. A new camera is one more `kind: video` rule. A rig that
also tracks the head becomes the 29-dim layout by adding a `kind: pose` rule for
it and listing that key in `state_layout` — no code change.

## Development

```bash
pip install -e .
python -m pytest tests/test_hdf52lerobot.py
```
