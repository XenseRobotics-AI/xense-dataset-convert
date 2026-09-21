# xense-dataset-convert

Robot dataset format converters, one package per direction. Built to be vendored
into a larger system as a git submodule, so all four live on `main` rather than
on separate branches.

| Package | Direction | CLI | Full docs |
|---|---|---|---|
| `lerobot2mcap` | LeRobot → MCAP | `lerobot2mcap` | [docs/lerobot2mcap.md](docs/lerobot2mcap.md) |
| `lerobot2zarr` | LeRobot → Zarr | `lerobot2zarr` | [docs/lerobot2zarr.md](docs/lerobot2zarr.md) |
| `mcap2lerobot` | MCAP → LeRobot v3 | `mcap2lerobot` | [docs/mcap2lerobot.md](docs/mcap2lerobot.md) |
| `hdf52lerobot` | UMI HDF5/MKV → LeRobot v3 | `hdf52lerobot` | [docs/hdf52lerobot.md](docs/hdf52lerobot.md) |

## Installation

One install provides all four commands:

```bash
git clone https://github.com/XenseRobotics-AI/xense-dataset-convert.git
cd xense-dataset-convert
pip install -e .
```

Dependencies are not split into extras, so this pulls `lerobot` and its
torch stack even if you only intend to use `hdf52lerobot` (whose own code never
imports lerobot — it writes the v3.0 layout directly).

### Standalone environment

The repository does not need the lerobot fork: any lerobot that reads and
writes the v3.0 dataset format works. The tested recipe is a conda environment
that provides Python and the `ffmpeg` binary (`lerobot2mcap` shells out to it),
with every Python dependency coming from `pyproject.toml`:

```bash
conda env create -f environment.yml   # python 3.12 + ffmpeg, then pip install -e .[dev]
conda activate xense-convert
pytest -q                             # 53 tests, no dataset needed
```

Without conda: Python >=3.12, an `ffmpeg` on `PATH`, then `pip install -e .[dev]`.
The `[dev]` extra adds `pytest` and `ruff`; everything else is a hard dependency.

### Which lerobot

`lerobot[dataset]>=0.5.1,<0.7`. The on-disk format is `v3.0` in every release
of that range; the cap is about lerobot's *Python API*, which three of the four
converters call:

| Package | lerobot symbols used |
|---|---|
| `lerobot2mcap` | `LeRobotDataset(...)`, for `download` |
| `lerobot2zarr` | `LeRobotDataset(...)`, to read frames |
| `mcap2lerobot` | `LeRobotDataset.create` / `add_frame` / `save_episode` / `finalize` |
| `hdf52lerobot` | none: it writes the v3.0 layout directly |

Both ends of the range were exercised end to end (every CLI plus the Python
facade, on a synthetic recording): 0.5.1 and 0.6.1. Bump the cap after testing
a new minor rather than leaving it open, because 0.6 already reorganised the
writer once. The `[dataset]` extra matters from 0.6 on, where `datasets`,
`pyarrow`, `pandas` and `torchcodec` left lerobot's core dependencies; 0.5.x
ignores it.

The [`xense-taccap-lerobot`](https://github.com/Vertax42/xense-taccap-lerobot)
fork (`0.5.1+xtac.0.0.6`) still satisfies the range, so installing into an
environment that already has it leaves it in place: PEP 440 sorts a local
version above the plain release.

## Unified Python interface

For calling from another system, `xense_dataset_convert` puts one entry point in
front of all four. The converter packages are untouched underneath; this layer
only reconciles interfaces that differ in shape, in what they accept, and in how
they report failure.

```python
from xense_dataset_convert import convert, available, Format

available()
# [(Format.LEROBOT, Format.MCAP), (Format.LEROBOT, Format.ZARR),
#  (Format.MCAP, Format.LEROBOT), (Format.UMI_HDF5, Format.LEROBOT)]

result = convert(Format.UMI_HDF5, Format.LEROBOT,
                 inputs="/data/SuHang", output="/data/SuHang_lerobot",
                 task="pick up the cup")
result.output_path     # Path("/data/SuHang_lerobot")
result.details         # {"total_episodes": 2, "total_frames": 1472}
```

`Format` is an enum; build one from a config value with `Format(cfg["src"])`.
Options are forwarded to the underlying converter, and an unrecognised name
raises with the accepted spellings rather than being ignored — these usually
come from a config file, where a typo would otherwise pass silently.

`available()` and a conversion that does not need lerobot import nothing that
does: the registry resolves adapters lazily, so an HDF5 → LeRobot run never
loads torch. `details` carries `total_episodes` / `total_frames` for LeRobot
targets, read back from `meta/info.json`; no converter returns counts directly,
so the key is absent rather than guessed when the metadata cannot be read.

One gap: `video_preset` and `video_crf` are not accepted for LeRobot → MCAP.
`lerobot2mcap` reads them from module-level globals rather than taking them as
arguments, so honouring them here would mean mutating process-wide state. Set
`lerobot2mcap.VIDEO_ENCODER_CRF` yourself, or use that package's CLI.

## The converters

### `lerobot2mcap` — LeRobot → MCAP

Converts LeRobot datasets to MCAP for Foxglove-style inspection, generating the
channel configuration automatically from the dataset's `info.json`. v3.0-first,
still backward compatible with v2.1/v2.x.

```bash
lerobot2mcap download lerobot/pusht
lerobot2mcap convert ~/.cache/huggingface/lerobot/pusht -o ./mcap_output
```

Custom per-topic converter functions live in `configs/converter_functions.yaml`.
See [docs/lerobot2mcap.md](docs/lerobot2mcap.md) for the `download` subcommand,
the required `info.json` fields, and the topic layout.

### `lerobot2zarr` — LeRobot → Zarr

Exports a LeRobot dataset to a generic, schema-described Zarr store, for
consumers that want array access without the LeRobot reader.

```bash
lerobot2zarr /path/to/lerobot/dataset -o ./dataset.zarr
```

See [docs/lerobot2zarr.md](docs/lerobot2zarr.md) for the output layout and the
schema rules that decide how each feature is stored.

### `mcap2lerobot` — MCAP → LeRobot v3

Converts ROS2, DAS-Ego, and merged DAS-UMI MCAP episode recordings into a LeRobot v3 dataset. DAS-UMI schema v3 adds full preflight reports, strict 30 Hz alignment, next-frame control targets, 0–1 normalized gripper opening, observation-only IMU/auxiliary poses, and an audit sidecar outside the training table.

```bash
mcap2lerobot ./mcap_output -o ./converted_dataset
```

See [docs/mcap2lerobot.md](docs/mcap2lerobot.md) for the generic topic layout, and
[mcap2lerobot/das_ego_UMI.md](mcap2lerobot/das_ego_UMI.md) for the DAS workflow and schema v3.

### `hdf52lerobot` — UMI HDF5/MKV → LeRobot v3

Converts the raw per-stream HDF5/MKV recordings from the bimanual handheld UMI
rig into a LeRobot v3.0 dataset, tagged `robot_type: bi_rdt_gripper`. No
`lerobot` install is needed at runtime: the layout is written directly and PyAV
does the transcoding.

```bash
hdf52lerobot --input /data/SuHang --output /data/SuHang_lerobot --task "your task"
```

Stream classification is config-driven — a new camera is one more rule in
[`configs/hdf52lerobot.yaml`](configs/hdf52lerobot.yaml), not a code change. See
[docs/hdf52lerobot.md](docs/hdf52lerobot.md) for the file-to-feature mapping,
the pairing modes and the rotation/gripper conventions.

## Converting the TacVerse-RDT corpus

The RDT capture team delivers recordings filed by capture station, with no task
text: `/data/data4/umi_data/<station>/ep_YYYYMMDD_HHMMSS/`. TacVerse datasets
are one task each, so the conversion regroups them and needs the team's task
schedule to do it. Three steps.

**1. Build the task map.** A recording's date and half-day decide its task, via
`采集任务时间安排表.xlsx`; [`configs/tacverse_rdt_catalogue.yaml`](configs/tacverse_rdt_catalogue.yaml)
turns each code into English task text and a directory slug.

```bash
python tools/rdt_task_map.py \
  --schedule ~/Documents/TacVerse-RDT-Rules/采集任务时间安排表.xlsx \
  --input /data/data4/umi_data/035 --input /data/data4/umi_data/036 \
  --output configs/tacverse_rdt_tasks.yaml
```

**2. See what will be produced.** One dataset per (task, date), merging every
station that recorded that task that day.

```bash
python tools/rdt_convert.py --list
```

Half-days that list several task codes are reported here and **skipped**: the
schedule says the session covered two or three tasks but not which recording is
which, so those episodes cannot be filed until the capture team supplies a
per-episode list. That is two thirds of the corpus.

**3. Convert.** `--jobs` is datasets at a time, `--workers` cameras within one:

```bash
python tools/rdt_convert.py --jobs 12 --workers 7 \
  --output /archive/lerobot/TacVerse/raw
python tools/rdt_progress.py --output /archive/lerobot/TacVerse/raw   # watch it
```

Twelve jobs convert about 22 hours of recording per hour on a 32-core machine
with an NVENC-capable GPU, so the 1245 hours that can be filed by task take
about two days. Converting is resumable twice over: a dataset with `meta/info.json` is left
alone, and a run stopped with Ctrl-C leaves a checkpoint the next run continues
from, to a result identical to never having stopped. A lock file keeps two
drivers off the same output tree.

### What the profile does to this rig's data

[`configs/tacverse_rdt.yaml`](configs/tacverse_rdt.yaml) is the profile, and
every setting in it answers something measured in the recordings:

| Setting | Why |
|---|---|
| `pair: grid`, `fps: 20` | The recorder's loop targets 30 Hz but stalls, averaging 20–24 Hz and varying per episode, so its iterations cannot be the time axis. Each episode is resampled onto a uniform grid. |
| `keep_source_timestamp: true` | Keeps the recorder's own time for each frame beside LeRobot's `frame_index / fps`. |
| `transform: [bgr_to_rgb, transpose]` on the tactile cameras | The rig recorded the Xense SDK's BGR bytes as if RGB, in the sensor's own 400×700 portrait frame. TacVerse stores tactile as 400×700 RGB. |
| `rgb_reference: GSPS01` | SDK 2.1.x shifts each sensor onto a reference tone at init; these recordings predate that, and the sensors' channel means are spread over some fifty levels. |
| `key: side` for the third camera | It is a fixed side view, not a headset camera, and the rig has no head tracker. |
| `vcodec: h264_nvenc`, `crf: 21` | Encoding was two thirds of the CPU work. QP 21 matched libx264 CRF 23 on PSNR (41.85 dB against 41.75) in a smaller file at 3.1× the speed. |
| `codec_threads: 1` | FFmpeg otherwise sizes every encoder and decoder to the whole machine: 1100 threads on 32 cores, and an episode taking 29.9 s instead of 19.4 s. |

Gripper angles need no setting: the encoder reads 0–360 and five of the 32
stations have a jaw straddling that wrap point, written as a negative
calibration bound (`openv2_-16.49_10.91`). Readings come to the equivalent
angle nearest the calibrated span before being normalised, which is identity
for the other 59 jaws.

## Development

```bash
pip install -e .[dev]
pytest -q                                           # lerobot2zarr, mcap2lerobot, hdf52lerobot, facade
ruff check --select E9,F63,F7,F82 .
```

`pyproject.toml` puts the repository root on `pythonpath` for pytest, so the
suite also runs against an uninstalled checkout, file by file or as a whole.

`lerobot2mcap` has no unit suite: `tests/` carries a standalone MCAP validator
(`tests/validate_mcap.py`) and an encoder benchmark (`tests/bench_encoder.sh`),
both of which need a real dataset on disk.
