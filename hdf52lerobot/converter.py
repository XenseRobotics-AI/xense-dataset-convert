"""Drive the whole conversion: recording directories in, v3.0 dataset out."""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
from dataclasses import asdict
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .checkpoint import CHECKPOINT_NAME, CHECKPOINT_VERSION, Checkpoint, InterruptRequest
from .config import ConverterConfig
from .features import SOURCE_TIMESTAMP, build_features, video_features
from .ee_transform import transform_pose_block
from .geometry import to_wxyz
from .logger import get_logger
from .source import SourceEpisode, read_episode
from .stats import StatsAccumulator, downsample_hwc, sample_positions
from .sync import SyncReport, build_sync
from .video import VideoAggregator
from .writer import DatasetWriter, VideoPlacement, seconds_to_ns
from xense_dataset_convert._parallelism import available_cpu_count

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


def warn_untransformed_poses(config: ConverterConfig, logger: Any) -> None:
    """Say so when an RDT conversion is about to write raw tracker poses.

    `bi_rdt_gripper` recordings carry the Vive tracker's pose under a key named
    `{side}_tcp`. Converting them with no `tracker_to_ee` is what produced
    every dataset of this rig so far, and it is silent -- the column names look
    right either way.
    """
    if config.robot_type != "bi_rdt_gripper":
        return
    missing = [
        key
        for key in config.state_layout
        if key.endswith("_tcp") and config.tracker_to_ee_for(key).is_identity
    ]
    if missing:
        logger.warn(
            f"{', '.join(missing)}: no tracker_to_ee entry, so the Vive tracker's own pose "
            f"is being written under a name that claims to be a TCP. See "
            f"hdf52lerobot/ee_transform.py."
        )
    elif not all(config.tracker_to_ee_for(key).measured for key in config.state_layout
                 if key.endswith("_tcp")):
        logger.warn(
            "tracker->EE offsets are marked provisional (measured: false); poses are placed "
            "at roughly the right scale but not to the millimetre."
        )


def build_state(episode: SourceEpisode, sync: SyncReport, config: ConverterConfig) -> np.ndarray:
    """Concatenate the configured streams into the ``observation.state`` matrix."""
    blocks: list[np.ndarray] = []
    for key in config.state_layout:
        indices = sync.indices_for(key)
        if key in episode.poses:
            values = episode.poses[key].values[indices]
            # The recorded frame is the tracker's, not the TCP the name claims;
            # ee_transform puts it where the name says and fixes the world.
            positions, rotations = transform_pose_block(
                values[:, :3],
                to_wxyz(values[:, 3:7], config.quat_order),
                config.tracker_to_ee_for(key),
                config.world_matrix(),
            )
            blocks.append(np.concatenate([positions, rotations], axis=1))
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
        skip_failed_episodes: bool = False,
        resume: bool = True,
    ):
        """
        Args:
            skip_failed_episodes: Carry on when a recording cannot be read,
                pairs badly, or does not match the others, instead of failing
                the run. Only failures raised before anything is written can be
                skipped, so a partly written episode still stops the run.
                Skipped recordings are listed in ``meta/sources.json``. Worth
                turning on for a bulk run over third-party recordings, where
                one damaged directory should not cost a dataset of thousands.
            resume: Carry on from the checkpoint an interrupted run left, when
                there is one for this configuration. Turn it off to insist on
                converting from the first episode.
        """
        self.inputs = list(inputs)
        self.output_root = Path(output_root).expanduser().resolve()
        self.config = config or ConverterConfig()
        self.overwrite = overwrite
        self.skip_failed_episodes = skip_failed_episodes
        self.resume = resume
        self._interrupt = InterruptRequest()

    def convert(self) -> Path:
        episode_paths = discover_episodes(self.inputs)
        logger.info(f"Converting {len(episode_paths)} episode(s) -> {self.output_root}")
        warn_untransformed_poses(self.config, logger)
        if self.config.world_frame != "none":
            logger.info(f"World frame: {self.config.world_frame}")

        resume = Checkpoint.load(self.output_root) if self.resume else None
        if resume is not None and resume.config_fingerprint != self._fingerprint():
            logger.warn(
                "checkpoint was written with a different configuration; starting over"
            )
            resume = None

        if self.output_root.exists():
            if resume is None:
                if not self.overwrite:
                    raise FileExistsError(
                        f"Output already exists: {self.output_root} (pass overwrite to replace it)"
                    )
                shutil.rmtree(self.output_root)
                self.output_root.mkdir(parents=True)
        else:
            self.output_root.mkdir(parents=True)

        done: set[str] = set()
        if resume is not None:
            done = set(resume.converted)
            logger.info(
                f"Resuming: {len(done)} episode(s) already converted, "
                f"{len(resume.skipped)} skipped"
            )

        skipped: list[dict[str, Any]] = list(resume.skipped) if resume else []
        first, first_position = self._first_readable(episode_paths, skipped)
        state_names = self.config.state_names(first.kinds)
        features = build_features(first, self.config, state_names)
        video_keys = video_features(features)

        writer = DatasetWriter(self.output_root, self.config, features, video_keys)
        aggregators = {
            stream.feature: VideoAggregator(
                self.output_root,
                stream.feature,
                stream.width,
                stream.height,
                self.config,
                stream.transform,
                stream.rgb_reference,
            )
            for stream in first.videos.values()
        }
        accumulator = StatsAccumulator()
        sources: list[dict[str, Any]] = list(resume.sources) if resume else []
        if resume is not None:
            writer.restore(resume.writer)
            accumulator.restore(resume.stats)
            for key, aggregator in aggregators.items():
                if key in resume.videos:
                    aggregator.restore(resume.videos[key])

        # One pool for the whole run: a pool per episode means fresh threads,
        # and every fresh thread can pull a new malloc arena that is never
        # returned, so memory would creep up with the episode count.
        workers = self.config.workers or max(1, available_cpu_count() // 2)
        pool = ThreadPoolExecutor(max_workers=min(workers, len(first.videos) or 1))
        interrupted = False
        try:
            with self._interrupt as interrupt:
                for position, path in enumerate(episode_paths):
                    if position < first_position:
                        continue  # already reported while looking for a reference
                    if str(path) in done:
                        continue  # converted before the interrupt
                    if interrupt.requested:
                        # Stop between episodes: mid-episode there would be a
                        # half-written mp4 and no row describing it.
                        interrupted = True
                        logger.warn(
                            f"Interrupted after {len(sources)} episode(s) this run; "
                            f"finishing the checkpoint"
                        )
                        break
                    try:
                        episode = (
                            first
                            if position == first_position
                            else read_episode(path, self.config)
                        )
                        self._check_layout(episode, first, state_names)
                        sync = build_sync(episode, self.config)
                    except Exception as error:
                        if not self.skip_failed_episodes:
                            raise
                        # Nothing has been written for this episode yet, so
                        # dropping it leaves the dataset consistent.
                        logger.warn(f"{path.name}: skipped ({type(error).__name__}: {error})")
                        skipped.append(self._skip_record(path, error))
                        continue
                    sources.append(
                        self._convert_episode(
                            episode, sync, writer, aggregators, accumulator, pool
                        )
                    )
        finally:
            pool.shutdown(wait=True)
            for aggregator in aggregators.values():
                aggregator.close()

        if interrupted:
            writer.flush()
            Checkpoint(
                version=CHECKPOINT_VERSION,
                config_fingerprint=self._fingerprint(),
                converted=sorted(done | {record["source_dir"] for record in sources}),
                skipped=skipped,
                sources=sources,
                writer=writer.state(),
                videos={key: aggregator.state() for key, aggregator in aggregators.items()},
                stats=accumulator.state(),
            ).save(self.output_root)
            raise KeyboardInterrupt(
                f"stopped after {len(sources)} episode(s); rerun to resume from "
                f"{self.output_root / CHECKPOINT_NAME}"
            )

        if not sources:
            raise ValueError(f"No episode could be converted; {len(skipped)} were skipped")
        writer.finalize(accumulator.finalize())
        self._write_sources(sources, skipped)
        Checkpoint.clear(self.output_root)
        if skipped:
            logger.warn(f"Skipped {len(skipped)} unusable recording(s); see meta/sources.json")
        logger.info(f"Done: {self.output_root}")
        return self.output_root

    def request_stop(self) -> None:
        """Ask a running conversion to stop after the episode in progress.

        What the signal handler does, available to a caller driving this in a
        thread. The run ends by raising ``KeyboardInterrupt`` once the
        checkpoint is written.
        """
        self._interrupt.requested = True

    def _fingerprint(self) -> str:
        """Identifies the configuration a checkpoint was written under.

        Resuming under different settings would splice two different datasets
        together -- a different fps, codec or stream layout partway through,
        with nothing in the result to show where the seam is.
        """
        payload = json.dumps(asdict(self.config), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @staticmethod
    def _skip_record(path: Path, error: Exception) -> dict[str, Any]:
        return {
            "source_dir": str(path),
            "source_parent": path.parent.name,
            "error": f"{type(error).__name__}: {error}",
        }

    def _first_readable(
        self, episode_paths: list[Path], skipped: list[dict[str, Any]]
    ) -> tuple[SourceEpisode, int]:
        """The episode the dataset's features are built from.

        Normally the first one. When skipping is on and it cannot be read, the
        search moves down the list rather than failing: the reference only has
        to be a valid recording, not the first one.
        """
        for position, path in enumerate(episode_paths):
            try:
                return read_episode(path, self.config), position
            except Exception as error:
                if not self.skip_failed_episodes:
                    raise
                logger.warn(f"{path.name}: skipped ({type(error).__name__}: {error})")
                skipped.append(self._skip_record(path, error))
        raise ValueError("No readable recording among the inputs")

    @staticmethod
    def _gripper_normalization(sources: list[dict[str, Any]]) -> dict[str, Any]:
        """What a normalized gripper value means, and the spread behind it.

        ``1.0`` is the open end of *this recording's calibration window*, not
        the jaw's mechanical limit. The two are not the same: the CTAG2F120 the
        RDT rig uses travels 40.1 degrees (a 2 to 88 mm pad gap), while the
        windows in the recordings span 15.5 to 29.9 degrees and differ per
        station and per session. A consumer that maps 1.0 onto the part's
        mechanical maximum -- a URDF's joint limit, say -- therefore draws a jaw
        roughly 40% wider open than the hardware ever was.

        So the recovery is per episode and lives with each episode's record;
        this block is the summary, for a consumer that wants the shape of the
        thing before reading 4000 rows.
        """
        travels: dict[str, list[float]] = {}
        for record in sources:
            for key, calibration in record.get("grippers", {}).items():
                travels.setdefault(key, []).append(calibration["travel_deg"])
        return {
            "definition": (
                "value = clip((angle - closed_deg) / (open_deg - closed_deg), 0, 1); "
                "0 is closed, 1 is the open end of this recording's calibration "
                "window, which is narrower than the jaw's mechanical travel"
            ),
            "per_episode_key": "episodes[].grippers",
            "travel_deg": {
                key: {
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "episodes": len(values),
                }
                for key, values in sorted(travels.items())
            },
        }

    def _write_sources(
        self, sources: list[dict[str, Any]], skipped: list[dict[str, Any]] | None = None
    ) -> None:
        """``meta/sources.json``: where every episode came from and how it was resampled.

        Not part of the LeRobot layout, which ignores extra files under meta/;
        it is what lets a downstream tool map an episode back to its recording.
        """
        from . import __version__

        config = dataclasses.asdict(self.config)
        config["streams"] = [dataclasses.asdict(rule) for rule in self.config.streams]
        document = {
            "format": "hdf52lerobot/sources",
            "format_version": 1,
            "converted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "hdf52lerobot_version": __version__,
            "inputs": [str(Path(entry).expanduser().resolve()) for entry in self.inputs],
            "config": config,
            "gripper_normalization": self._gripper_normalization(sources),
            "episodes": sources,
            "skipped": skipped or [],
        }
        path = self.output_root / "meta" / "sources.json"
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
        sync: SyncReport,
        writer: DatasetWriter,
        aggregators: dict[str, VideoAggregator],
        accumulator: StatsAccumulator,
        pool: ThreadPoolExecutor,
    ) -> dict[str, Any]:
        """Write one episode; returns its ``meta/sources.json`` entry."""
        logger.info(f"{episode.name}: {sync.length} frames @ {self.config.fps}fps")
        for message in sync.unchecked:
            logger.warn(f"{episode.name}: {message} -- not checked, so kept")
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
        futures = {
            stream.feature: pool.submit(
                aggregators[stream.feature].append,
                stream.path,
                sync.indices_for(key),
                positions,
                downsample_hwc,
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

        task = self.config.task_for(episode.name)
        record = writer.add_episode(
            states,
            actions,
            task,
            placements,
            source_timestamps=sync.tick_timestamps if self.config.keep_source_timestamp else None,
        )
        length = record.length
        # The same columns the parquet rows carry, so the per-episode block and
        # stats.json describe exactly what was written.
        numeric = {
            "observation.state": states,
            "action": actions,
            "timestamp": np.arange(length) / self.config.fps,
            "frame_index": np.arange(length),
            "episode_index": np.full(length, record.episode_index),
            "index": np.arange(record.dataset_from_index, record.dataset_to_index),
            "task_index": np.full(length, writer.task_index(task)),
        }
        if self.config.keep_source_timestamp:
            numeric[SOURCE_TIMESTAMP] = seconds_to_ns(sync.tick_timestamps)
        record.stats = accumulator.add_episode(numeric=numeric, images=samples)

        offsets_ms = np.abs(sync.warp_s) * 1000.0
        worst_gap_ms = {
            key: round(float(stream.gaps_s.max()) * 1000, 1)
            for key, stream in sync.streams.items()
        }
        loose_frames = {key: stream.loose for key, stream in sync.streams.items() if stream.loose}
        return {
            "episode_index": record.episode_index,
            "source_dir": str(episode.path),
            "source_parent": episode.path.parent.name,
            "task": task,
            "frames": length,
            "source_iterations": sync.source_length,
            "pair": sync.mode,
            "duplicated_frames": sync.duplicated if sync.mode == "grid" else 0,
            "dropped_iterations": sync.dropped if sync.mode == "grid" else 0,
            "tick_offset_ms": {
                "p50": round(float(np.percentile(offsets_ms, 50)), 1),
                "p95": round(float(np.percentile(offsets_ms, 95)), 1),
                "max": round(float(offsets_ms.max()), 1),
            },
            # What a quality pass downstream filters on: how stale the worst
            # frame of each stream is, and how many sat past the warn limit.
            "worst_gap_ms": max(worst_gap_ms.values()),
            "worst_gap_stream": max(worst_gap_ms, key=worst_gap_ms.get),
            "loose_frames": loose_frames,
            "first_source_timestamp": float(sync.tick_timestamps[0]),
            "last_source_timestamp": float(sync.tick_timestamps[-1]),
            "files": sorted(stream.path.name for stream in episode.all_streams().values()),
            "grippers": {
                key: {
                    "closed_deg": round(gripper.lo, 4),
                    "open_deg": round(gripper.hi, 4),
                    "travel_deg": round(gripper.hi - gripper.lo, 4),
                    "travel_rad": round(math.radians(gripper.hi - gripper.lo), 6),
                    "wraps_zero": gripper.lo < 0.0,
                }
                for key, gripper in sorted(episode.grippers.items())
            },
        }


__all__ = [
    "Hdf5ToLeRobotConverter",
    "discover_episodes",
    "build_state",
    "build_action",
]
