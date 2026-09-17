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
as `observation.imu`, `.left_wrist`, and `.right_wrist`. Encoder readings keep
their source value as `observation.magnetic.<side>.raw_value`; their physical
unit has not been verified and they are not presented as normalized jaw width.
Configured `PoseInFrame` streams are exported as XYZ plus the first two columns
of the rotation matrix (`r1` … `r6`), without a coordinate-frame transform.
World and relative poses remain separate features, and the head block of state
uses the world pose.

`observation.state` follows the TacVerse order, omitting unavailable optional
blocks rather than filling them with zeros:

```text
left hand pose, left raw encoder, right hand pose, right raw encoder, head pose
```

With the default merged capture (no hand-pose topics) this is 11 values: two
raw encoder values and a 9D head pose. Configuring both optional hand poses
makes it 29 values. By default no action is fabricated. With `--export-action`,
`action[t] = observation.state[t+1]`; the last row is dropped, dimensions and
names match state, and pairing never crosses an episode boundary.

`observation.source_timestamp` contains the target grid time. Every exported
stream also has `observation.source_timestamp.<stream>` and
`observation.source_index.<stream>`, recording the actual selected acquisition
time and original MCAP message ordinal.

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

Export observations without action (the default):

```bash
mcap2lerobot output/59214a9e/merged_output_ego_vio.mcap \
  -o ./59214a9e_lerobot --config configs/das_umi.yaml --no-export-action
```

Export observations plus one-step-ahead action:

```bash
mcap2lerobot output/59214a9e/merged_output_ego_vio.mcap \
  -o ./59214a9e_lerobot_action --export-action \
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
    export_action=True,
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
