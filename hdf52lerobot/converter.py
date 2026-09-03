"""Drive the whole conversion: recording directories in, v3.0 dataset out."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import ConverterConfig
from .features import build_features, video_features
from .geometry import quaternions_to_rotation_6d, to_wxyz
from .logger import get_logger
from .source import SourceEpisode, read_episode
from .stats import StatsAccumulator, sample_positions
from .sync import SyncReport, build_sync
from .video import VideoAggregator
from .writer import DatasetWriter, VideoPlacement

logger = get_logger("hdf52lerobot")


def discover_episodes(inputs: Sequence[str | Path]) -> list[Path]:
    """Resolve inputs to episode directories.

    An input is treated as an episode when it directly contains recorder files,
    and as a parent directory otherwise (its sub-directories become episodes,
    sorted by name so the episode order is the recording order).
    """
    episodes: list[Path] = []
    for entry in inputs:
        path = Path(entry).expanduser().resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Input is not a directory: {path}")
        if any(path.glob("*.hdf5")) or any(path.glob("*.mkv")):
            episodes.append(path)
            continue
        children = sorted(child for child in path.iterdir() if child.is_dir())
        if not children:
            raise ValueError(f"{path}: contains neither recorder files nor episode directories")
        episodes.extend(children)

    unique = list(dict.fromkeys(episodes))
    if not unique:
        raise ValueError("No episode directories found")
    return unique


def build_state(episode: SourceEpisode, sync: SyncReport, config: ConverterConfig) -> np.ndarray:
    """Concatenate the configured streams into the ``observation.state`` matrix."""
    blocks: list[np.ndarray] = []
    for key in config.state_layout:
        indices = sync.indices_for(key)
        if key in episode.poses:
            values = episode.poses[key].values[indices]
            rotations = quaternions_to_rotation_6d(to_wxyz(values[:, 3:7], config.quat_order))
            blocks.append(np.concatenate([values[:, :3], rotations], axis=1))
        elif key in episode.grippers:
            blocks.append(episode.grippers[key].normalized()[indices][:, None])
        else:
            raise ValueError(f"{episode.name}: {key!r} is neither a pose nor a gripper stream")
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_action(states: np.ndarray) -> np.ndarray:
    """``action[t] = state[t+1]``; the final frame repeats its own state.

    Matches TacCap, where every episode satisfies ``action[t] == state[t+1]``.
    The recorder there had a real action for the last frame; we have no next
    state, so the last action holds.
    """
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


class Hdf5ToLeRobotConverter:
    """Convert one or more recording directories into a single v3.0 dataset."""

    def __init__(
        self,
        inputs: Sequence[str | Path],
        output_root: str | Path,
        config: ConverterConfig | None = None,
        overwrite: bool = False,
    ):
        self.inputs = list(inputs)
        self.output_root = Path(output_root).expanduser().resolve()
        self.config = config or ConverterConfig()
        self.overwrite = overwrite

    def convert(self) -> Path:
        episode_paths = discover_episodes(self.inputs)
        logger.info(f"Converting {len(episode_paths)} episode(s) -> {self.output_root}")

        if self.output_root.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Output already exists: {self.output_root} (pass overwrite to replace it)"
                )
            shutil.rmtree(self.output_root)
        self.output_root.mkdir(parents=True)

        first = read_episode(episode_paths[0], self.config)
        state_names = self.config.state_names(first.kinds)
        features = build_features(first, self.config, state_names)
        video_keys = video_features(features)

        writer = DatasetWriter(self.output_root, self.config, features, video_keys)
        aggregators = {
            stream.feature: VideoAggregator(
                self.output_root, stream.feature, stream.width, stream.height, self.config
            )
            for stream in first.videos.values()
        }
        accumulator = StatsAccumulator()

        try:
            for position, path in enumerate(episode_paths):
                episode = first if position == 0 else read_episode(path, self.config)
                self._check_layout(episode, first, state_names)
                self._convert_episode(episode, writer, aggregators, accumulator)
        finally:
            for aggregator in aggregators.values():
                aggregator.close()

        writer.finalize(accumulator.finalize())
        logger.info(f"Done: {self.output_root}")
        return self.output_root

    # -- internals -----------------------------------------------------------

    def _check_layout(
        self, episode: SourceEpisode, reference: SourceEpisode, state_names: list[str]
    ) -> None:
        if set(episode.videos) != set(reference.videos):
            raise ValueError(
                f"{episode.name}: camera set {sorted(episode.videos)} differs from "
                f"{reference.name}'s {sorted(reference.videos)}"
            )
        for key, stream in episode.videos.items():
            other = reference.videos[key]
            if (stream.width, stream.height) != (other.width, other.height):
                raise ValueError(
                    f"{episode.name}: {key} is {stream.width}x{stream.height}, "
                    f"expected {other.width}x{other.height}"
                )
        if self.config.state_names(episode.kinds) != state_names:
            raise ValueError(f"{episode.name}: state layout differs from {reference.name}")

    def _convert_episode(
        self,
        episode: SourceEpisode,
        writer: DatasetWriter,
        aggregators: dict[str, VideoAggregator],
        accumulator: StatsAccumulator,
    ) -> None:
        sync = build_sync(episode, self.config)
        logger.info(f"{episode.name}: {sync.length} frames @ {self.config.fps}fps")
        for line in sync.describe():
            logger.info(f"  {line}")
        loose = {key: stream.loose for key, stream in sync.streams.items() if stream.loose}
        if loose:
            logger.warn(
                f"{episode.name}: {sum(loose.values())} frame(s) served by a sample further than "
                f"{self.config.warn_sync_gap_frames} frames from their tick (recorder hiccup): {loose}"
            )

        states = build_state(episode, sync, self.config)
        actions = build_action(states)
        positions = sample_positions(sync.length)

        placements: dict[str, VideoPlacement] = {}
        samples: dict[str, list[np.ndarray]] = {}
        workers = self.config.workers or max(1, (os.cpu_count() or 2) // 2)
        with ThreadPoolExecutor(max_workers=min(workers, len(episode.videos) or 1)) as pool:
            futures = {
                stream.feature: pool.submit(
                    aggregators[stream.feature].append,
                    stream.path,
                    sync.indices_for(key),
                    positions,
                )
                for key, stream in episode.videos.items()
            }
            for feature, future in futures.items():
                result = future.result()
                placements[feature] = VideoPlacement(
                    chunk_index=result.chunk_index,
                    file_index=result.file_index,
                    from_timestamp=result.from_timestamp,
                    to_timestamp=result.to_timestamp,
                )
                samples[feature] = result.samples

        record = writer.add_episode(states, actions, self.config.task, placements)
        length = record.length
        # The same columns the parquet rows carry, so the per-episode block and
        # stats.json describe exactly what was written.
        record.stats = accumulator.add_episode(
            numeric={
                "observation.state": states,
                "action": actions,
                "timestamp": np.arange(length) / self.config.fps,
                "frame_index": np.arange(length),
                "episode_index": np.full(length, record.episode_index),
                "index": np.arange(record.dataset_from_index, record.dataset_to_index),
                "task_index": np.full(length, writer.task_index(self.config.task)),
            },
            images=samples,
        )


__all__ = [
    "Hdf5ToLeRobotConverter",
    "discover_episodes",
    "build_state",
    "build_action",
]
