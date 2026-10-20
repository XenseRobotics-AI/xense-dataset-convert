# lerobot2mcap

Convert LeRobot datasets to MCAP format with automatic configuration generation from dataset metadata. No manual configuration required – just point it at your dataset and go. **This version is primarily focused on LeRobot v3.0 merged-format datasets** (parquet+video merged, per-episode slicing), while remaining backward compatible with v2.1/v2.x. AV1 is supported via a PyAV monkey‑patch.

## Features (v3.0-first)

- **Automatic configuration**: Reads `meta/info.json` and generates all necessary configuration using Pydantic models for type safety
- **Episode-based conversion**: Converts each episode to a separate MCAP file in its own directory
- **Chunk-aware**: Handles datasets organized in chunks (supports datasets with 1000+ episodes)
- **Multi-video support**: Auto-detects and converts all video streams
- **Round-trip metadata**: every episode MCAP carries `meta/info.json` (the dataset's metadata, verbatim) and `meta/episode.json` (episode index, length, task texts) as attachments, which `mcap2lerobot` reads back to restore names, dtypes, fps, robot type, codec and tasks
- **Terminal log support**: a `.log` file in the dataset root is carried into every episode's MCAP as an attachment
- **ROS2 format**: Outputs ROS2-compatible MCAP files (configurable via metadata)
- **v3.0-first**: Handles merged parquet+video with per-episode slicing (rows + timestamp-based video trim)
- **Cross-compatible**: Still supports LeRobot v2.0, v2.1 (per-episode files)
- **All-keyframe video**: each episode's video is sliced and re-encoded with ffmpeg as all-keyframe H.264, so every `CompressedVideo` message decodes on its own
- **Metadata-driven**: Reads FPS, video codecs, writer format, and chunk size from dataset metadata
- **Per-episode parallelism**: `-j` sets how many cameras of one episode are sliced and encoded concurrently; episodes themselves run sequentially

## Installation

```bash
# Clone the repository
git clone https://github.com/XenseRobotics-AI/xense-dataset-convert.git
cd xense-dataset-convert

# Install in editable mode
pip install -e .

# Open help menu
lerobot2mcap --help
```

## Requirements

Your LeRobot dataset should have the following structure:

```
dataset_root/
├── meta/
│   └── info.json          # Required: Dataset metadata
├── data/                  # Required: Parquet data files
└── videos/                # Optional: Video files (if dataset contains video)
```

### Required `info.json` Fields

The converter reads the following fields from `meta/info.json`:

| Field              | Type   | Description                                                  | Default                      |
| ------------------ | ------ | ------------------------------------------------------------ | ---------------------------- |
| `codebase_version` | string | Dataset format version (e.g., "v2.0", "v3.0")                | _(required)_                 |
| `total_episodes`   | int    | Total number of episodes in the dataset                      | _(required)_                 |
| `fps`              | int    | Frames per second for the dataset                            | _(required)_                 |
| `features`         | dict   | Feature definitions including video streams and their codecs | _(required)_                 |
| `data_path`        | string | Template path for parquet files                              | _(required)_                 |
| `video_path`       | string | Template path for video files                                | _(required if videos exist)_ |
| `chunks_size`      | int    | Maximum episodes per chunk                                   | 1000                         |
| `writer_format`    | string | MCAP writer format ("ros1", "ros2", "json", "protobuf")      | "ros2"                       |

### Video Codec Support

- Default encode: `h264` (configurable via `DEFAULT_CODEC` in `dataset_info.py`)
- Decode: PyAV monkey-patch (supports AV1/H.264/…); replaces OpenCV loader in tabular2mcap

## Architecture

The converter uses an object-oriented, metadata-driven architecture:

```
┌─────────────────┐
│  DatasetInfo    │  Parses meta/info.json to extract:
│                 │  • total_episodes, chunks_size, fps
└────────┬────────┘  • video_keys and codecs from features
         │           • data_path and video_path templates
         │           • codebase_version, writer_format
         ↓
┌─────────────────┐
│ ConfigGenerator │  Generates per-episode Pydantic configs:
│                 │  • TabularMappingConfig (parquet data)
└────────┬────────┘  • CompressedVideoMappingConfig (videos)
         │           • AttachmentConfig (log files)
         │           • Validates all fields at runtime
         ↓
┌─────────────────┐
│ LeRobotConverter│  Orchestrates conversion:
│                 │  • Iterates through episodes
└────────┬────────┘  • Calls tabular2mcap per episode
         │           • Saves config alongside MCAP
         ↓
┌─────────────────┐
│  tabular2mcap   │  Performs actual MCAP writing:
│                 │  • Reads parquet, videos, logs
└─────────────────┘  • Writes MCAP with ROS2 schemas
```

**Key Design Principles (v3.0-first):**

1. **Type Safety**: Uses Pydantic models from `tabular2mcap.loader.models` for validation
2. **Template-Based**: Builds configs from a base template with `.model_copy()`
3. **Metadata-Driven**: Reads all values from `info.json` instead of hardcoding
4. **Episode-Level**: Processes one episode at a time for predictable output structure
5. **v3.0 merging aware**: Splits merged parquet rows and trims merged videos per episode

## Usage

The tool provides two main commands: `download` and `convert`.

### Quick Start

```bash
# Download and convert in one command (downloads from Hugging Face)
lerobot2mcap download lerobot/pusht

# Or convert an existing dataset
lerobot2mcap convert ~/.cache/huggingface/lerobot/pusht -o ./mcap_output
```

### Command: `download`

Download LeRobot datasets from Hugging Face Hub and automatically convert to MCAP.

```bash
# Download full dataset (uses default HF cache, outputs to ./lerobot_pusht_mcap/)
lerobot2mcap download lerobot/pusht

# Download with custom MCAP output directory
lerobot2mcap download lerobot/pusht -o ./my_mcap_output

# Download specific episodes only
lerobot2mcap download lerobot/pusht -e 0 1 2

# Encode up to 8 cameras of an episode concurrently
lerobot2mcap download lerobot/pusht -j8
```

**Arguments:**

- `dataset_id`: Hugging Face dataset ID (e.g., `lerobot/pusht`)
- `-o, --output-dir`: Output directory for MCAP files (default: `./{dataset_name}_mcap`)
- `-e, --episodes`: Specific episode indices to download (default: all episodes)
- `-j, --jobs`: Cameras sliced and encoded concurrently within one episode
  (default: min(8, available CPU count // 2), with a minimum of 1)

**Output:**

- Downloads dataset to HuggingFace cache (`~/.cache/huggingface/lerobot/`)
- Creates MCAP files in specified output directory (or `./{dataset_name}_mcap`)

### Command: `convert`

Convert an existing LeRobot dataset to MCAP format.

```bash
# Convert all episodes (configuration auto-generated)
lerobot2mcap convert ~/.cache/huggingface/lerobot/pusht -o ./mcap_output

# Convert specific episodes only
lerobot2mcap convert /path/to/dataset -o ./mcap_output -e 0 1 2

# Encode up to 8 cameras of an episode concurrently
lerobot2mcap convert /path/to/dataset -o ./mcap_output -j8

# Custom converter functions (advanced)
lerobot2mcap convert /path/to/dataset -o ./mcap_output -f ./my_converter_functions.yaml
```

**Arguments:**

- `input_dir`: Path to LeRobot dataset root (must contain `meta/info.json`)
- `-o, --output-dir`: Output directory for MCAP files (default: `<input_dir>/mcap_conversion`)
- `-e, --episodes`: Specific episode indices to convert (default: all episodes)
- `-j, --jobs`: Cameras sliced and encoded concurrently within one episode
  (default: min(8, available CPU count // 2), with a minimum of 1)
- `-f, --converter-functions`: Path to custom converter functions YAML (default: built-in `configs/converter_functions.yaml`)

**Performance (50 episodes, 16-core CPU):**

| Workers | Time | Speedup |
| ------- | ---- | ------- |
| 1       | 77s  | 1.0x    |
| 4       | 48s  | 1.6x    |
| 8       | 45s  | 1.7x    |

**What Happens During Conversion (v3.0-first):**

1. Reads `meta/info.json` to understand dataset structure (merged files in v3.0)
2. Calculates total chunks (`ceil(total_episodes / chunks_size)`)—ignored for v3.0 merged files
3. For each episode, sequentially (`-j` parallelises the cameras inside it):
   - Slices parquet rows by episode index range (v3.0)
   - Slices video by timestamps per episode (v3.0) using ffmpeg, re-encoded as all-keyframe H.264
   - Writes temp files:
     - `robot/actions.parquet` → `/robot/actions/data`
     - `robot/states.parquet` → `/robot/states/data`
     - `observation/images/{cam}.h264` → `/observation/images/{cam}/video`
   - Generates temp config (Pydantic) and calls `tabular2mcap` with `strip_file_suffix=True`
   - Saves MCAP directly to output directory as `episode_XXX.mcap`
4. After all episodes finish, saves a single `config_{output_dir_name}.yaml` in the output directory
5. Skips episodes that fail (warned and listed in the summary) and reports failure, so a partial output is never mistaken for a complete one

### Topic layout (v3.0 example)

- `/robot/actions/data` — action vectors
- `/robot/states/data` — observation.state vectors
- `/observation/images/{cam}/video` — compressed video (foxglove_msgs/CompressedVideo)

### Recording Your Own Dataset With Terminal Logs

To capture terminal logs during recording (optional but recommended for debugging):

```bash
# Set log file path
LOG_FILE=~/.cache/huggingface/lerobot/${HF_USER}/my-dataset.log

# Record with lerobot-record and capture terminal output
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem5A680114161 \
    --robot.id=my_follower_arm \
    --robot.cameras="{
        front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30},
        external: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 15}
    }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem5A680123701 \
    --teleop.id=my_leader_arm \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/my-dataset \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=0 \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick and place demo" \
    2>&1 | tee $LOG_FILE

# Move log file to dataset root for automatic inclusion
mv $LOG_FILE ~/.cache/huggingface/lerobot/${HF_USER}/my-dataset/recording.log
```

**Note:** Place the `.log` file in the dataset root directory. The converter will automatically:

- Detect it using `**/*.log` pattern
- Parse it into `rcl_interfaces/msg/Log` messages
- Include it as an attachment in the MCAP file

### Expected Dataset Structure

The converter automatically detects LeRobot v3.0 (its primary target) as well as v2.0 and v2.1 from `meta/info.json`. Example input file structures are given below for v3 and v2.1 (very similar to v2).

#### LeRobot v3 Format

In LeRobot v3.0 several episodes are concatenated into the same files, rolling over at the parquet and MP4 size limits recorded in `info.json`. The structure below is an example; how many episodes share a file depends on the recording. The converter reads `meta/episodes/` to find each episode's row range and video timestamp range and writes one MCAP per episode.

```
dataset_root/
├── meta/
│   ├── info.json           # Dataset metadata (includes total_episodes)
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet  # per-episode row ranges and video timestamp ranges
├── data/
│   └── chunk-000/
│       ├── file-000.parquet  # Episode 0-3
│       ├── file-001.parquet  # Episode 4-5
│       └── file-002.parquet  # Episode 6-9
├── videos/
│   ├── observation.images.front/
│   │   └── chunk-000/
│   │       ├── file-000.mp4  # Episode 0-3
│   │       ├── file-001.mp4  # Episode 4-5
│   │       └── file-002.mp4  # Episode 6-9
│   └── observation.images.external/
│       └── chunk-000/
│           ├── file-000.mp4  # Episode 0-3
│           ├── file-001.mp4  # Episode 4-5
│           └── file-002.mp4  # Episode 6-9
└── recording.log           # Optional terminal log
```

#### LeRobot v2.1 Format

Episodes use 6-digit indices (episode_000000, episode_000001, etc.).

```
dataset_root/
├── meta/
│   ├── info.json           # Dataset metadata (includes total_episodes)
│   ├── episodes.jsonl
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # Episode 0
│       ├── episode_000001.parquet  # Episode 1
│       └── episode_000002.parquet  # Episode 2
└── videos/
    └── chunk-000/
        ├── observation.images.phone/
        │   ├── episode_000000.mp4  # Episode 0
        │   ├── episode_000001.mp4  # Episode 1
        │   └── episode_000002.mp4  # Episode 2
        └── observation.images.external/
            ├── episode_000000.mp4  # Episode 0
            ├── episode_000001.mp4  # Episode 1
            └── episode_000002.mp4  # Episode 2
```

**Note**: The converter automatically detects which format your dataset uses from `meta/info.json` and handles it appropriately.

## Output

All episodes are saved directly in the output directory with a single shared configuration file:

```
mcap_output/
├── config_mcap_output.yaml   # Single config file (named after output dir)
├── episode_000.mcap
├── episode_001.mcap
└── episode_002.mcap
```

Each MCAP file contains:

- **Robot data** from the episode's parquet rows (topics: `/robot/actions/data`, `/robot/states/data`)
- **Video streams** for each camera (topics: `/observation/images/{camera}/video`)
- **Source metadata** as attachments: `meta/info.json` and, for v3.0 datasets, `meta/episode.json`
- **Terminal log** as an MCAP attachment when a `.log` file sits in the dataset root

## How Configuration Works

**You don't need to create any configuration files** - everything is automatic:

1. **DatasetInfo** reads your `meta/info.json` to discover:

   - Total number of episodes
   - Video streams and their codecs
   - Dataset FPS and structure
   - Data file patterns

2. **ConfigGenerator** creates a config for each episode:

   - Tabular mappings pointing to the specific episode's parquet file
   - Video mappings pointing to the specific episode's video files
   - An attachment mapping for a `.log` file in the dataset root
   - One config is saved next to the MCAP files, named after the output directory

3. **tabular2mcap** uses each generated config to convert the episode to MCAP

### Advanced: Custom Converter Functions (Optional)

By default, the converter uses [`configs/converter_functions.yaml`](configs/converter_functions.yaml) for data transformation. You can customize this with the `-f` flag:

```yaml
# Custom converter_functions.yaml
functions:
  row_to_message_with_timestamp:
    schema_name: null
    template: |
      {
        "timestamp": {
          "sec": {{ (timestamp) | int }},
          "nanosec": {{ ((timestamp % 1) * 1_000_000_000) | int }}
        }
      }
```

**Note**: Log parsing is handled automatically by tabular2mcap's `LogConverter` - no converter function needed.

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e .

# Run tests
python -m pytest

# Build package
python -m build
```

## Browse Datasets

https://huggingface.co/lerobot
