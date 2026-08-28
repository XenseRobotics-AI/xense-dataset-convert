"""Orchestrate MCAP episode conversion into a LeRobot v3 dataset."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .config import ConverterConfig
from .reader import EpisodeData, McapEpisodeReader, TimedArray, TimedImage

logger = logging.getLogger(__name__)


class DatasetWriter(Protocol):
    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def finalize(self) -> None: ...


def discover_mcap_files(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        path = Path(input_path).expanduser().resolve()
        if path.is_dir():
            files.extend(path.glob("*.mcap"))
        elif path.is_file() and path.suffix.lower() == ".mcap":
            files.append(path)
        else:
            raise FileNotFoundError(f"MCAP input does not exist: {path}")
    unique_files = list(dict.fromkeys(files))
    return sorted(unique_files, key=_episode_sort_key)


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 2**63 - 1, path.name)


def infer_fps(episode: EpisodeData) -> int:
    timestamps = np.asarray([item.timestamp_ns for item in episode.states], dtype=np.int64)
    if timestamps.size < 2:
        raise ValueError("Cannot infer fps from an episode with fewer than two states")
    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        raise ValueError("Cannot infer fps from duplicate state timestamps")
    fps = int(round(1_000_000_000 / float(np.median(intervals))))
    if fps <= 0:
        raise ValueError("Inferred fps is invalid")
    return fps


def nearest_value(
    values: Sequence[TimedArray] | Sequence[TimedImage],
    timestamp_ns: int,
    max_gap_ns: int,
    label: str,
) -> np.ndarray:
    timestamps = np.asarray([item.timestamp_ns for item in values], dtype=np.int64)
    insertion = int(np.searchsorted(timestamps, timestamp_ns))
    candidates = []
    if insertion < len(values):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    if not candidates:
        raise ValueError(f"No samples available for {label}")
    index = min(candidates, key=lambda candidate: abs(int(timestamps[candidate]) - timestamp_ns))
    gap = abs(int(timestamps[index]) - timestamp_ns)
    if gap > max_gap_ns:
        raise ValueError(
            f"Nearest {label} sample is {gap / 1e6:.2f} ms away, "
            f"exceeding the {max_gap_ns / 1e6:.2f} ms limit"
        )
    return values[index].value


def build_features(episode: EpisodeData) -> dict[str, dict[str, Any]]:
    state = episode.states[0].value
    action = episode.actions[0].value
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": state.dtype.name,
            "shape": state.shape,
            "names": [f"state_{index}" for index in range(state.size)],
        },
        "action": {
            "dtype": action.dtype.name,
            "shape": action.shape,
            "names": [f"action_{index}" for index in range(action.size)],
        },
    }
    for feature_name, images in sorted(episode.videos.items()):
        image = images[0].value
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            raise ValueError(f"{feature_name} must be an HWC image, got {image.shape}")
        features[feature_name] = {
            "dtype": "video",
            "shape": image.shape,
            "names": ["height", "width", "channels"],
        }
    return features


class McapToLeRobotConverter:
    """Convert one MCAP file per episode into one LeRobot v3 dataset."""

    def __init__(
        self,
        inputs: Sequence[Path],
        output_root: Path,
        repo_id: str,
        config: ConverterConfig | None = None,
        reader: McapEpisodeReader | None = None,
        dataset_factory: Callable[..., DatasetWriter] | None = None,
    ):
        self.inputs = [Path(path) for path in inputs]
        self.output_root = Path(output_root).expanduser().resolve()
        self.repo_id = repo_id
        self.config = config or ConverterConfig()
        self.reader = reader or McapEpisodeReader(self.config)
        self.dataset_factory = dataset_factory or self._default_dataset_factory

    @staticmethod
    def _default_dataset_factory(**kwargs: Any) -> DatasetWriter:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise ImportError("lerobot==0.4.1 is required to write the dataset") from error
        return LeRobotDataset.create(**kwargs)

    def convert(self) -> Path:
        mcap_files = discover_mcap_files(self.inputs)
        if not mcap_files:
            raise ValueError("No .mcap files found")
        if self.output_root.exists():
            raise FileExistsError(f"Output path already exists: {self.output_root}")

        first_episode = self.reader.read(mcap_files[0])
        fps = self.config.fps or infer_fps(first_episode)
        features = build_features(first_episode)
        logger.info("Creating LeRobot v3 dataset at %s (fps=%d)", self.output_root, fps)
        dataset = self.dataset_factory(
            repo_id=self.repo_id,
            root=self.output_root,
            fps=fps,
            robot_type=self.config.robot_type,
            features=features,
            use_videos=bool(first_episode.videos),
        )

        try:
            self._write_episode(dataset, first_episode, features, fps)
            for mcap_path in mcap_files[1:]:
                episode = self.reader.read(mcap_path)
                self._write_episode(dataset, episode, features, fps)
            dataset.finalize()
        except Exception:
            finalize = getattr(dataset, "finalize", None)
            if callable(finalize):
                try:
                    finalize()
                except Exception:
                    pass
            raise
        return self.output_root

    def _write_episode(
        self,
        dataset: DatasetWriter,
        episode: EpisodeData,
        expected_features: dict[str, dict[str, Any]],
        fps: int,
    ) -> None:
        logger.info("Converting episode from %s", episode.source)
        actual_features = build_features(episode)
        if actual_features != expected_features:
            raise ValueError(f"Feature layout changed in {episode.source}")

        max_gap_ns = int(self.config.max_sync_gap_frames * 1_000_000_000 / fps)
        for frame_index, state in enumerate(episode.states):
            frame: dict[str, Any] = {
                "observation.state": state.value.copy(),
                "action": nearest_value(
                    episode.actions, state.timestamp_ns, max_gap_ns, "action"
                ).copy(),
                "timestamp": frame_index / fps,
                "task": self.config.task,
            }
            for feature_name, images in episode.videos.items():
                frame[feature_name] = nearest_value(
                    images, state.timestamp_ns, max_gap_ns, feature_name
                ).copy()
            dataset.add_frame(frame)
        dataset.save_episode()
        logger.info("Saved episode with %d frames", len(episode.states))
