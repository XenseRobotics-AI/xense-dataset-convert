# mcap2lerobot

Convert MCAP episode files into a LeRobot v3.0 dataset.

Two source layouts (dialects) are supported, chosen with `--dialect`:

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

## `lerobot_roundtrip` dialect

Select with `--dialect lerobot_roundtrip`. The default topics are:

| MCAP topic | LeRobot feature |
| --- | --- |
| `robot/states/data` | `observation.state` |
| `robot/actions/data` | `action` |
| `observation/images/<camera>/video` | `observation.images.<camera>` |

The state message must contain an `observation_state` numeric array and the
action message must contain an `action` numeric array. Video topics must use the
ROS2 `foxglove_msgs/msg/CompressedVideo` schema with independently decodable
H.264, H.265, VP9, or AV1 packets.

## Synchronization (both dialects)

The state stream is the output timeline. Actions and camera frames are matched
to each state timestamp using nearest-neighbor synchronization with a bounded
maximum gap. The pairing is decided before anything is written, so an episode
that is out of sync fails without leaving a half-written dataset.

Reading is done in two passes so memory does not grow with episode length: the
first pass keeps the state and action vectors plus each camera's packet
timestamps, the second decodes only the packets the pairing needs and hands
them to lerobot one state at a time. Design notes, with measurements, are in
[`mcap2lerobot/README.md`](../mcap2lerobot/README.md).

In the `lerobot_roundtrip` dialect, MCAP files written by `lerobot2mcap` also
carry two attachments, `meta/info.json` (the source dataset's metadata,
verbatim) and `meta/episode.json` (the episode's index, length and task texts).
When they are present the converter restores what the ROS messages alone cannot
say: the `observation.state` / `action` component names and dtype, `fps`,
`robot_type`, the video codec, and each frame's task text. `--fps`, `--task` and
`--robot-type` override them (in either dialect); an MCAP without the
attachments gets the generated names and the defaults listed under limitations.

## Installation

```bash
pip install -e .
```

The dataset is written through lerobot's own `LeRobotDataset` API, which
encodes video with PyAV; no `ffmpeg` binary is needed for this direction.

## Usage

Convert a directory of XTacData recordings (the default dialect):

```bash
mcap2lerobot /path/to/XTacData \
  --output ./converted_dataset \
  --repo-id local/converted_dataset \
  --task "Pick up the blocks one by one and place them inside the box."
```

Convert selected files:

```bash
mcap2lerobot episode_000.mcap episode_001.mcap -o ./converted_dataset
```

Convert `lerobot2mcap` round-trip output instead:

```bash
mcap2lerobot ./mcap_output -o ./converted_dataset \
  --dialect lerobot_roundtrip \
  --state-topic robot/states/data \
  --action-topic robot/actions/data \
  --state-field observation_state \
  --action-field action
```

Run `mcap2lerobot --help` for all options.

## Current limitations

- Two source layouts are supported (`--dialect`): `xtac` (JSON + protobuf) and
  `lerobot_roundtrip` (ROS2 CDR). Other topic layouts are not.
- One MCAP file must represent exactly one episode.
- State and action features are one-dimensional numeric arrays.
- For MCAP files without the `meta/info.json` / `meta/episode.json`
  attachments (written by `lerobot2mcap` before 0.3.2, or by other tools), task
  text and robot type come from CLI options or the defaults `Converted from
  MCAP` / `mcap`, vector names are generated as `state_0`, `action_0`, and so
  on, and lerobot encodes video with its own default codec.
- Per-episode statistics are recomputed by lerobot; `meta/stats.json` of the
  source is not carried over.
- The output directory must not already exist.
