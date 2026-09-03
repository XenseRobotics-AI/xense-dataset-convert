# mcap2lerobot

Convert ROS2 MCAP episode files into a LeRobot v3.0 dataset.

This converter targets the MCAP files written by this repository's
`lerobot2mcap`. It intentionally does not try to support every robot or
arbitrary ROS topic layout yet.

## Supported MCAP layout

Each `.mcap` file is treated as one episode. The default topics are:

| MCAP topic | LeRobot feature |
| --- | --- |
| `robot/states/data` | `observation.state` |
| `robot/actions/data` | `action` |
| `observation/images/<camera>/video` | `observation.images.<camera>` |

The state message must contain an `observation_state` numeric array and the
action message must contain an `action` numeric array. Video topics must use the
ROS2 `foxglove_msgs/msg/CompressedVideo` schema with independently decodable
H.264, H.265, VP9, or AV1 packets.

The state stream is the output timeline. Actions and camera frames are matched
to each state timestamp using nearest-neighbor synchronization with a bounded
maximum gap. The pairing is decided before anything is written, so an episode
that is out of sync fails without leaving a half-written dataset.

Reading is done in two passes so memory does not grow with episode length: the
first pass keeps the state and action vectors plus each camera's packet
timestamps, the second decodes only the packets the pairing needs and hands
them to lerobot one state at a time. Design notes, with measurements, are in
[`mcap2lerobot/README.md`](../mcap2lerobot/README.md).

MCAP files written by `lerobot2mcap` also carry two attachments,
`meta/info.json` (the source dataset's metadata, verbatim) and
`meta/episode.json` (the episode's index, length and task texts). When they are
present the converter restores what the ROS messages alone cannot say: the
`observation.state` / `action` component names and dtype, `fps`, `robot_type`,
the video codec, and each frame's task text. `--fps`, `--task` and
`--robot-type` override them; an MCAP without the attachments gets the
generated names and the defaults listed under limitations.

## Installation

```bash
pip install -e .
```

The dataset is written through lerobot's own `LeRobotDataset` API, which
encodes video with PyAV; no `ffmpeg` binary is needed for this direction.

## Usage

Convert all MCAP files in a directory:

```bash
mcap2lerobot ./mcap_output \
  --output ./converted_dataset \
  --repo-id local/converted_dataset \
  --task "Pick up the object"
```

Convert selected files:

```bash
mcap2lerobot episode_000.mcap episode_001.mcap -o ./converted_dataset
```

FPS is inferred from state timestamps. Override it when timestamps are noisy:

```bash
mcap2lerobot ./mcap_output -o ./converted_dataset --fps 30
```

Topic and field overrides are available for small variations:

```bash
mcap2lerobot ./mcap_output -o ./converted_dataset \
  --state-topic robot/states/data \
  --action-topic robot/actions/data \
  --state-field observation_state \
  --action-field action
```

Run `mcap2lerobot --help` for all options.

## Current limitations

- Only ROS2 CDR MCAP messages are decoded.
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
