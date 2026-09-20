# mcap2lerobot

Convert XTacData, ROS2 round-trip, DAS-Ego, or merged DAS-UMI MCAP episodes
into a LeRobot v3.0 dataset. DAS profiles are detected before the dialect
fallback; non-DAS inputs use `--dialect` (`xtac` by default or
`lerobot_roundtrip`). The converter does not support arbitrary ROS layouts.

| Dialect | Source | Default |
| --- | --- | --- |
| `xtac` | XTacData recordings from the XTac-UMI-G1 grippers | ✅ |
| `lerobot_roundtrip` | the ROS2 CDR layout this repository's `lerobot2mcap` emits | |

Each `.mcap` file is one episode in both dialects.

## `xtac` dialect (default)

XTacData mixes JSON and protobuf on one recording. The `/train/state` and
`/train/action` topics are JSON (`taccap.train.State/Action`); their 29-value
`values` array is the `observation.state` / `action` vector. The eight cameras
are protobuf `foxglove.CompressedVideo` streams of inter-frame H.264:

| MCAP topic | LeRobot feature |
| --- | --- |
| `/camera/head/left/h264` | `observation.images.head_left` |
| `/camera/head/right/h264` | `observation.images.head_right` |
| `/camera/left/fisheye/h264` | `observation.images.left_wrist` |
| `/camera/right/fisheye/h264` | `observation.images.right_wrist` |
| `/camera/left/tactile_1/h264` | `observation.images.left_tactile_left` |
| `/camera/left/tactile_2/h264` | `observation.images.left_tactile_right` |
| `/camera/right/tactile_1/h264` | `observation.images.right_tactile_left` |
| `/camera/right/tactile_2/h264` | `observation.images.right_tactile_right` |

`fps`, the vector component names, `robot_type` (`xtac_umi_g1`) and the
per-camera calibration come from the MCAP **metadata records**
(`xense_train_mcap`, `episode`), not attachments. `observation.tracker_confidence`
(the PICO drift warning) is read from `/pico/pose` and paired to each state.

After the dataset is written, the extra provenance files the reference TacVerse
datasets carry are reproduced from the same metadata: `meta/hardware.json`,
`meta/xumi_collection_devices.json`, `meta/taccap_extrinsics.json`, and the
tactile cameras' `taccap.rectify_config` merged into `meta/info.json`. The
`tracker_to_ee` transform is left null: the recorded poses are already the
end-effector (TCP).

The task text has no human-readable default in the source (only a `task_id`), so
pass `--task` for a real instruction.

## Profiles and `lerobot_roundtrip` dialect

`--profile auto` is the default. A capture containing any `robot1/` or
`robot2/` Finger topic is classified as `das-umi` before the older `das-ego`
check. A partial Finger recording therefore fails DAS-UMI preflight instead of
silently becoming an Ego-only dataset. Non-DAS inputs fall back to the selected
dialect.

Select `--dialect lerobot_roundtrip` for MCAP files produced by this repository
from `lerobot2mcap`. Its default topics are:
| MCAP topic | LeRobot feature |
| --- | --- |
| `robot/states/data` | `observation.state` |
| `robot/actions/data` | `action` |
| `observation/images/<camera>/video` | `observation.images.<camera>` |

The state stream is its output timeline. Actions and images use bounded nearest
neighbours. Embedded `meta/info.json` and `meta/episode.json` attachments restore
feature names/dtypes, fps, robot type, codec, and task text when available.

## XTac and round-trip synchronization

The state stream is the output timeline. Actions and camera frames are matched
to each state timestamp using nearest-neighbor synchronization with a bounded
maximum gap. Pairing is decided before anything is written. Round-trip MCAPs
can restore feature names, dtypes, fps, robot type, codec, and task text from
embedded `meta/info.json` and `meta/episode.json` attachments.

The compatible `das-ego` profile reads the six Ego cameras and Ego IMU. The
strict `das-umi` profile reads one merged Ego + left Finger + right Finger file;
its topic map and quality thresholds live in
[`configs/das_umi.yaml`](../configs/das_umi.yaml).

## DAS-UMI output

DAS-UMI always uses this exact integer-nanosecond 30 Hz grid over the common
usable interval of every exported continuous stream:
```text
T[k] = start + floor(k * 1,000,000,000 / 30)
```

The eight video features are the existing
`observation.images.camera0` … `camera5` plus
`observation.images.left_wrist` and `right_wrist`. The three IMUs are exported
as `observation.imu`, `.left_wrist`, and `.right_wrist`. World and relative
head poses remain separate XYZ + 6D rotation features. Rotation uses matrix
columns `[R00,R10,R20,R01,R11,R21]`, without a coordinate-frame transform.

By default, `observation.state` and `action` are float32 vectors with these
same 11 names, in order:

| Indices | Names | Source |
|---|---|---|
| 0 | `left_gripper.pos` | Left encoder |
| 1 | `right_gripper.pos` | Right encoder |
| 2–4 | `head_camera.x`, `.y`, `.z` | Head world position |
| 5–10 | `head_camera.r1` … `.r6` | Head world rotation |

Encoder values pass through unchanged except for float32 conversion. They are
provisionally interpreted as radians: **units unconfirmed / 单位待确认**, unlike
TacCap normalized gripper opening. No normalization or jaw-distance conversion
is applied. Duplicate `observation.magnetic.left/right.raw_value` columns are
no longer exported; encoder samples remain available internally for alignment
and their source timestamp/index columns are retained. Matching field names
does not imply calibration to the TacCap camera coordinate frame.

For each episode, first align the complete sequence `S[0…N-1]`, then export:

```text
observation.state = S[0…N-2]
action            = S[1…N-1]
```

Action is the next-frame **absolute state**, not a delta. The final observation
row, including its images and auxiliary fields, is dropped because it has no
next-frame target. Episodes with fewer than two aligned rows fail preflight.
Use `--no-export-action` (Python: `export_action=False`) to retain all aligned
rows as observations only. Omitted/`None`, explicit `True`, and explicit `False`
are handled consistently by CLI, Python facade, and `ConverterConfig`; the
default-on rule applies only to DAS-UMI.

No hand TCP is generated, no external service is called, and missing pose
blocks are not filled with zeros. Existing optional hand-pose configuration
remains supported (29 values with both hand poses). The eight images, three
IMUs, and independent head poses do not extend the default training vector.

`observation.source_timestamp` contains the derived target grid time. Every
sampled stream also has `observation.source_timestamp.<stream>` and
`observation.source_index.<stream>`, recording its current observation row's
selected acquisition time and original MCAP message ordinal.
`meta/das_umi_topic_mapping.json` lists each actual source topic's exported
fields, audit columns, and state/action component names and indices; action
components explicitly use the next aligned frame. Disabling action removes
its mapping entries. The mapping identifies grid time as derived, with no
source topic. Calibration remains in `meta/das_umi_calibration.json`; preflight
reports list unconverted topics. LeRobot v3 data/video path templates, device
type, camera views, and image resolutions are unchanged.

Conversion reports include the vector layout, action setting, tail row count,
gripper unit assumption, mapping path, and validation results (action validation
is N/A when disabled).

## DAS-UMI preflight and reports

Preflight scans the entire MCAP and decodes every video packet without retaining
the whole video in memory. It reports topic disposition, raw/usable ranges,
duration, counts, measured rate, interval P50/P95/P99/MAX, sequence gaps,
decode errors, clock offsets, nearest-neighbour P50/P95/P99/MAX, and
selected/unselected/repeated samples.

Acquisition time comes from `header.timestamp`. A stream with no valid header
time uses `log_time` for the entire stream and says so in the report; partial
per-message fallback is rejected. Header differences from both `log_time` and
`publish_time` are checked. Required missing streams, non-finite data, invalid
poses/quaternions, duplicate/backward time, sequence loss, mid-stream video
corruption, resolution changes, excessive periods, endpoints, or synchronization
gaps reject the batch. An undecodable video prefix of at most one second may be
trimmed; its first decodable packet is retained.

All files in a batch must pass and have the same feature layout before the
writer is created. `preflight.md` and `conversion.md` default to the sibling
directory `<output-name>_reports/`; a rejected preflight leaves both reports and
creates no LeRobot dataset. Calibration topics are additionally retained in
`meta/das_umi_calibration.json`; audio, system status, depth, and other
unconfigured topics are inventoried with an exclusion reason but not copied.

A real conversion writes a sibling temporary dataset, then checks LeRobot
loading, Parquet row counts and integer timelines, all output-video frame counts
and resolutions, and the shifted-action equality. Only a successful result is
atomically moved to the requested path. A failed post-write validation leaves
the temporary path in `conversion.md` and never overwrites an existing dataset.

## Installation

```bash
pip install -e .[dev]
```

The LeRobot writer uses PyAV; no `ffmpeg` binary is required in this direction.

## Usage

Convert a directory of XTacData recordings (the default dialect):

```bash
mcap2lerobot /path/to/XTacData \
  --output ./converted_dataset \
  --repo-id local/converted_dataset \
  --task "Pick up the blocks one by one and place them inside the box."
```

Preflight only (reports are still written):

```bash
mcap2lerobot output/59214a9e/merged_output_ego_vio.mcap \
  -o ./59214a9e_lerobot --profile das-umi --check-only
```

Export observations without action:

```bash
mcap2lerobot output/59214a9e/merged_output_ego_vio.mcap \
  -o ./59214a9e_lerobot --config configs/das_umi.yaml --no-export-action
```

Export the default training observations and next-frame absolute action:

```bash
mcap2lerobot output/59214a9e/merged_output_ego_vio.mcap \
  -o ./59214a9e_lerobot_action --profile das-umi \
  --report-dir ./59214a9e_reports
```

Convert `lerobot2mcap` round-trip output and optionally override its topics:

```bash
mcap2lerobot ./mcap_output -o ./converted_dataset \
  --dialect lerobot_roundtrip \
  --state-topic robot/states/data \
  --action-topic robot/actions/data \
  --state-field observation_state \
  --action-field action
```

Run `mcap2lerobot --help` for all options.

The unified Python facade accepts the same options:

```python
from xense_dataset_convert import Format, convert

convert(
    Format.MCAP,
    Format.LEROBOT,
    inputs="output/59214a9e/merged_output_ego_vio.mcap",
    output="59214a9e_lerobot",
    profile="das-umi",
    config="configs/das_umi.yaml",
    # export_action=False keeps all rows as observations only.
)
```

Existing ROS2 and DAS-Ego commands remain valid:

```bash
mcap2lerobot ./mcap_output -o ./converted_dataset --dialect lerobot_roundtrip --task "Pick up the object"
mcap2lerobot ego.mcap -o ./ego_dataset --profile das-ego --fps 30
```

Run `mcap2lerobot --help` for topic/field overrides and all options.

## Limits

- Supported source layouts are XTacData (`--dialect xtac`), ROS2 round-trip
  (`--dialect lerobot_roundtrip`), DAS-Ego, and merged DAS-UMI.
- Round-trip state/action features are one-dimensional numeric arrays.
- Round-trip MCAP files without embedded metadata use generated names and CLI defaults.
- Protobuf DAS support covers the configured Foxglove/DAS schemas; it is not a
  generic arbitrary-topic converter.
- The output directory must not already exist.
- Per-episode LeRobot statistics are recomputed; source `meta/stats.json` is not
  copied.
