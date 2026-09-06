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

Converts ROS2 MCAP episode recordings into a LeRobot v3 dataset.

```bash
mcap2lerobot ./mcap_output -o ./converted_dataset
```

See [docs/mcap2lerobot.md](docs/mcap2lerobot.md) for the expected topic layout
and current limitations.

### `hdf52lerobot` — UMI HDF5/MKV → LeRobot v3

Converts the raw per-stream HDF5/MKV recordings from the bimanual handheld UMI
rig into a LeRobot v3.0 dataset, tagged `robot_type: bi_rdt_gripper`. No
`lerobot` install is needed at runtime: the layout is written directly and PyAV
does the transcoding.

```bash
hdf52lerobot --input /data/SuHang --output /data/SuHang_lerobot --task "your task"
```

Stream classification is config-driven — a new camera is one more rule in
[`configs/hdf52lerobot.yaml`](configs/hdf52lerobot.yaml), not a code change.
[`configs/tacverse_rdt.yaml`](configs/tacverse_rdt.yaml) is the profile for the
TacVerse-RDT recordings, whose recorder loop does not run at a fixed rate: it
resamples every episode onto a 20 Hz grid (`pair: grid`) and keeps the
recorder's own timestamps as `observation.source_timestamp`. See
[docs/hdf52lerobot.md](docs/hdf52lerobot.md) for the file-to-feature mapping,
the pairing modes and the rotation/gripper conventions.

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
