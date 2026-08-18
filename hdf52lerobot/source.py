"""Scan one recording directory and classify its files into semantic streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import GRIPPER, POSE, VIDEO, ConverterConfig
from .logger import get_logger

logger = get_logger("hdf52lerobot.source")

SIDECAR_SUFFIX = ".timestamp.hdf5"
"""Per-frame timestamps that accompany a video file."""

FLOAT_DATASET = "float_array"
TIMESTAMP_DATASET = "timestamp_array"


@dataclass
class PoseStream:
    """A 6-DoF tracker: ``[x, y, z]`` plus a quaternion, one row per frame."""

    key: str
    path: Path
    timestamps: np.ndarray
    values: np.ndarray
    kind: str = POSE


@dataclass
class GripperStream:
    """A jaw encoder with its calibration bounds parsed from the file name."""

    key: str
    path: Path
    timestamps: np.ndarray
    values: np.ndarray
    lo: float
    hi: float
    kind: str = GRIPPER

    def normalized(self) -> np.ndarray:
        """Map raw encoder counts to ``[0, 1]``, 0 = closed, 1 = open."""
        span = self.hi - self.lo
        if span == 0:
            raise ValueError(f"{self.path.name}: degenerate gripper calibration [{self.lo}, {self.hi}]")
        return np.clip((self.values - self.lo) / span, 0.0, 1.0)


@dataclass
class VideoStream:
    """A camera: an encoded video plus its per-frame timestamp sidecar."""

    key: str
    path: Path
    timestamps: np.ndarray
    width: int
    height: int
    kind: str = VIDEO

    @property
    def feature(self) -> str:
        return f"observation.images.{self.key}"


@dataclass
class SourceEpisode:
    """Every stream found in one recording directory."""

    path: Path
    poses: dict[str, PoseStream] = field(default_factory=dict)
    grippers: dict[str, GripperStream] = field(default_factory=dict)
    videos: dict[str, VideoStream] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def kinds(self) -> dict[str, str]:
        """Stream key -> kind, for every non-video stream plus the videos."""
        kinds = {key: POSE for key in self.poses}
        kinds.update({key: GRIPPER for key in self.grippers})
        kinds.update({key: VIDEO for key in self.videos})
        return kinds

    def timestamps_of(self, key: str) -> np.ndarray:
        for group in (self.poses, self.grippers, self.videos):
            if key in group:
                return group[key].timestamps
        raise KeyError(f"{self.name}: no stream named {key!r}")

    def all_streams(self) -> dict[str, PoseStream | GripperStream | VideoStream]:
        merged: dict[str, PoseStream | GripperStream | VideoStream] = {}
        merged.update(self.poses)
        merged.update(self.grippers)
        merged.update(self.videos)
        return merged


def _read_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(values, timestamps)`` from one recorder HDF5 file."""
    import h5py

    with h5py.File(path, "r") as handle:
        for dataset in (FLOAT_DATASET, TIMESTAMP_DATASET):
            if dataset not in handle:
                raise ValueError(f"{path.name}: missing dataset {dataset!r}")
        values = np.asarray(handle[FLOAT_DATASET])
        timestamps = np.asarray(handle[TIMESTAMP_DATASET], dtype=np.float64)

    values = np.squeeze(values)
    if values.ndim == 0:
        values = values.reshape(1)
    if len(values) != len(timestamps):
        raise ValueError(
            f"{path.name}: {len(values)} values but {len(timestamps)} timestamps"
        )
    return values, timestamps


def _read_sidecar(video_path: Path) -> np.ndarray:
    sidecar = video_path.with_name(video_path.name + SIDECAR_SUFFIX)
    if not sidecar.exists():
        raise FileNotFoundError(f"{video_path.name}: missing timestamp sidecar {sidecar.name}")
    _, timestamps = _read_hdf5(sidecar)
    return timestamps


def _probe_video(path: Path) -> tuple[int, int]:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path.name}: contains no video stream")
        codec_context = container.streams.video[0].codec_context
        return int(codec_context.width), int(codec_context.height)


def read_episode(path: str | Path, config: ConverterConfig) -> SourceEpisode:
    """Classify every file in ``path`` according to the config's stream rules.

    Files that match no rule are logged and skipped -- never silently dropped.
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Not an episode directory: {directory}")

    episode = SourceEpisode(path=directory)
    unmatched: list[str] = []

    for entry in sorted(directory.iterdir()):
        if not entry.is_file() or entry.name.endswith(SIDECAR_SUFFIX):
            continue

        matched = False
        for rule in config.streams:
            resolved = rule.match(entry.name)
            if resolved is None:
                continue
            key, groups = resolved
            _add_stream(episode, rule.kind, key, entry, groups)
            matched = True
            break
        if not matched:
            unmatched.append(entry.name)

    if unmatched:
        logger.warn(f"{episode.name}: {len(unmatched)} file(s) matched no stream rule: {unmatched}")
    _validate(episode, config)
    return episode


def _add_stream(
    episode: SourceEpisode,
    kind: str,
    key: str,
    path: Path,
    groups: dict[str, str],
) -> None:
    existing = episode.all_streams().get(key)
    if existing is not None:
        raise ValueError(
            f"{episode.name}: stream key {key!r} claimed by both "
            f"{existing.path.name} and {path.name}"
        )

    if kind == VIDEO:
        timestamps = _read_sidecar(path)
        width, height = _probe_video(path)
        episode.videos[key] = VideoStream(
            key=key, path=path, timestamps=timestamps, width=width, height=height
        )
        return

    values, timestamps = _read_hdf5(path)
    if kind == POSE:
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError(f"{path.name}: expected pose shape (N, 7), got {values.shape}")
        episode.poses[key] = PoseStream(
            key=key, path=path, timestamps=timestamps, values=values.astype(np.float64)
        )
        return

    if values.ndim != 1:
        raise ValueError(f"{path.name}: expected gripper shape (N,), got {values.shape}")
    try:
        lo, hi = float(groups["lo"]), float(groups["hi"])
    except KeyError as error:
        raise ValueError(
            f"{path.name}: gripper rule must capture 'lo' and 'hi' calibration groups"
        ) from error
    episode.grippers[key] = GripperStream(
        key=key, path=path, timestamps=timestamps, values=values.astype(np.float64), lo=lo, hi=hi
    )


def _validate(episode: SourceEpisode, config: ConverterConfig) -> None:
    streams = episode.all_streams()
    if not streams:
        raise ValueError(f"{episode.name}: no recognized streams")
    if config.master_clock not in streams:
        raise ValueError(
            f"{episode.name}: master clock {config.master_clock!r} not found "
            f"(available: {sorted(streams)})"
        )
    for key in config.state_layout:
        if key not in streams:
            raise ValueError(f"{episode.name}: state_layout needs {key!r}, which is missing")

    for key, stream in streams.items():
        timestamps = stream.timestamps
        if len(timestamps) == 0:
            raise ValueError(f"{episode.name}: stream {key!r} is empty")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError(f"{episode.name}: stream {key!r} has non-monotonic timestamps")

    for key, gripper in episode.grippers.items():
        outside = np.count_nonzero(
            (gripper.values < gripper.lo - 1e-9) | (gripper.values > gripper.hi + 1e-9)
        )
        if outside:
            fraction = 100.0 * outside / len(gripper.values)
            logger.warn(
                f"{episode.name}: {key} has {outside} sample(s) ({fraction:.1f}%) outside the "
                f"calibrated range [{gripper.lo}, {gripper.hi}]; they will be clipped"
            )
