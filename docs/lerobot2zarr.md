# lerobot2zarr

Convert LeRobot datasets to a generic, schema-described Zarr store. The
converter discovers frame-aligned features from the dataset instead of
assuming a fixed camera layout, robot type, or number of cameras.

## Features

- Supports LeRobot datasets readable by `LeRobotDataset`.
- Preserves every frame-aligned feature, including action, state, timestamps,
  task fields, and any number of image/video features.
- Maps feature names such as `observation.images.head_left` to
  `data/observation/images/head_left`.
- Normalizes image/video frames to HWC `uint8`.
- Retains numeric and boolean feature shapes and dtypes.
- Records the original schema and source metadata in `meta/schema.json`.
- Records exclusive episode boundaries in `meta/episode_ends` and original
  episode IDs in `meta/episode_ids`.
- Supports selecting episodes and limiting frame count for quick validation.

## Installation

This implementation currently lives on the `feature/lerobot2zarr` branch of
the `XenseRobotics-AI/lerobot2mcap` GitHub repository. The Python package and
CLI are named `lerobot2zarr`; pushing this branch does not rename the GitHub
repository.

```bash
git clone --branch feature/lerobot2zarr \
  https://github.com/XenseRobotics-AI/lerobot2mcap.git
cd lerobot2mcap
python -m pip install -e .
lerobot2zarr --help
```

If the repository is already cloned, switch to the feature branch instead:

```bash
cd lerobot2mcap
git fetch origin
git switch feature/lerobot2zarr
python -m pip install -e .
```

## Usage

Convert a local LeRobot dataset:

```bash
lerobot2zarr /path/to/lerobot/dataset -o ./dataset.zarr
```

Select episodes or convert only a small prefix for validation:

```bash
lerobot2zarr /path/to/lerobot/dataset \
  --episodes 0 2 5 \
  --limit 100 \
  --output ./dataset_preview.zarr
```

If `--output` is omitted, the converter writes
`<input_dir>/zarr_conversion.zarr`.

For datasets whose local directory name does not match the LeRobot repository
ID, pass `--repo-id`:

```bash
lerobot2zarr /path/to/dataset \
  --repo-id organization/dataset \
  --output ./dataset.zarr
```

## Output Layout

```text
dataset.zarr/
├── data/
│   ├── action
│   ├── observation/
│   │   ├── state
│   │   └── images/
│   │       ├── head_left
│   │       └── head_right
│   └── timestamp
└── meta/
    ├── episode_ends
    ├── episode_ids
    └── schema.json
```

The actual feature list is determined by the input dataset. A dataset with no
camera features is valid; a dataset with one, two, or many cameras is also
valid as long as every selected frame has a fixed shape for each feature.

`meta/episode_ends` stores exclusive frame offsets. For example, values
`[120, 245]` mean episode one occupies frames `[0, 120)` and episode two
occupies frames `[120, 245)`.

## Schema Rules

The converter uses `meta/info.json` as optional feature metadata. Feature names
are preserved in `meta/schema.json`, while dots are converted to nested Zarr
groups. The schema records:

- Original feature name
- Zarr path
- Per-frame shape
- Stored dtype
- Feature kind (`numeric`, `string`, `image`, or `video`)
- Source dtype metadata
- Any normalization applied

Image and video features may be CHW or HWC. They are stored as contiguous HWC
`uint8` arrays. Floating-point images in `[0, 1]` are scaled to `[0, 255]`.
Numeric and boolean values are stored without semantic transformations.

All selected frames must contain every discovered feature, and each feature
must keep the same shape throughout the conversion. The converter fails early
on missing features, shape changes, unsupported values, or non-contiguous
episode ordering.

## Development

Run the focused test suite:

```bash
python -m pytest -q tests/test_zarr_converter.py
```

Compile the package:

```bash
python -m py_compile lerobot2zarr/*.py tests/test_zarr_converter.py
```
