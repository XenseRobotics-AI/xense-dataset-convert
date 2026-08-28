# mcap2lerobot

Convert ROS2 MCAP episode files into a LeRobot v3.0 dataset.

This branch currently targets MCAP files produced by this repository's former
`lerobot2mcap` implementation. It intentionally does not try to support every
robot or arbitrary ROS topic layout yet.

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
maximum gap.

## Installation

```bash
pip install -e .
```

`ffmpeg` must be available because LeRobot uses it to encode the generated v3
video files.

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
- Task text, robot type, and joint names cannot be recovered from the current
  MCAP files, so task and robot type come from CLI options and vector names are
  generated as `state_0`, `action_0`, and so on.
- The output directory must not already exist.

Embedding the original LeRobot feature metadata in future MCAP files would make
the conversion fully reversible, including joint names and task metadata.
