# xense-data-convert

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
git clone https://github.com/XenseRobotics-AI/lerobot2mcap.git
cd lerobot2mcap
pip install -e .
```

Dependencies are not split into extras, so this pulls `lerobot==0.4.1` and its
torch stack even if you only intend to use `hdf52lerobot` (whose own code never
imports lerobot — it writes the v3.0 layout directly).

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
rig into a LeRobot v3.0 dataset, tagged `robot_type: bi_rtc_gripper`. No
`lerobot` install is needed at runtime: the layout is written directly and PyAV
does the transcoding.

```bash
hdf52lerobot --input /data/SuHang --output /data/SuHang_lerobot --task "your task"
```

Stream classification is config-driven — a new camera is one more rule in
[`configs/hdf52lerobot.yaml`](configs/hdf52lerobot.yaml), not a code change.
See [docs/hdf52lerobot.md](docs/hdf52lerobot.md) for the file-to-feature mapping
and the rotation/gripper conventions.

## Development

```bash
pytest tests/ -q                                    # lerobot2zarr, mcap2lerobot, hdf52lerobot
ruff check --select E9,F63,F7,F82 .
```

`lerobot2mcap` has no unit suite: `tests/` carries a standalone MCAP validator
(`tests/validate_mcap.py`) and an encoder benchmark (`tests/bench_encoder.sh`),
both of which need a real dataset on disk.
