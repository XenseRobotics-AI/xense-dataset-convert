"""Write the LeRobot v3.0 on-disk layout.

Structurally identical to ``/home/xense-5090d/data/TacCap``: frames from every
episode are concatenated into rolling parquet files, videos into rolling mp4
files, and ``meta/episodes`` records where each episode lives.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import ConverterConfig
from .features import SOURCE_TIMESTAMP

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


@dataclass
class VideoPlacement:
    """Where one episode's frames sit inside one camera key's video files."""

    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass
class EpisodeRecord:
    """One row of ``meta/episodes``."""

    episode_index: int
    tasks: list[str]
    length: int
    data_chunk_index: int
    data_file_index: int
    dataset_from_index: int
    dataset_to_index: int
    videos: dict[str, VideoPlacement] = field(default_factory=dict)
    meta_chunk_index: int = 0
    meta_file_index: int = 0
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    """This episode's own ``{feature: {min, max, mean, std, count}}``, stored as
    ``stats/<feature>/<stat>`` columns so LeRobot's dataset tools can
    re-aggregate after deleting, splitting or merging episodes."""


class DatasetWriter:
    """Accumulate frames and metadata, then flush the whole v3.0 tree."""

    def __init__(
        self,
        root: Path,
        config: ConverterConfig,
        features: dict[str, dict[str, Any]],
        video_keys: list[str],
    ):
        self.root = Path(root)
        self.config = config
        self.features = features
        self.video_keys = list(video_keys)

        self.state_dim = len(features["observation.state"]["names"])
        self.source_timestamp = SOURCE_TIMESTAMP in features
        self._buffer: list[dict[str, np.ndarray]] = []
        self._buffer_rows = 0
        self._data_chunk_index = 0
        self._data_file_index = 0

        self._episodes: list[EpisodeRecord] = []
        self._tasks: list[str] = []
        self._total_frames = 0

    # -- tasks ---------------------------------------------------------------

    def task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks.append(task)
        return self._tasks.index(task)

    # -- episodes ------------------------------------------------------------

    def add_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        task: str,
        videos: dict[str, VideoPlacement],
        source_timestamps: np.ndarray | None = None,
    ) -> EpisodeRecord:
        """Append one episode's frames; returns its ``meta/episodes`` row.

        Args:
            source_timestamps: Absolute recorder time per frame in seconds,
                required when the features carry ``observation.source_timestamp``;
                stored as int64 nanoseconds.
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.shape != actions.shape:
            raise ValueError(f"state/action shape mismatch: {states.shape} vs {actions.shape}")
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError(f"expected states of shape (N, {self.state_dim}), got {states.shape}")

        length = len(states)
        episode_index = len(self._episodes)
        self._roll_data_file_if_needed()

        columns = {
            "observation.state": states,
            "action": actions,
            "timestamp": (np.arange(length) / self.config.fps).astype(np.float32),
            "frame_index": np.arange(length, dtype=np.int64),
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(self._total_frames, self._total_frames + length, dtype=np.int64),
            "task_index": np.full(length, self.task_index(task), dtype=np.int64),
        }
        if self.source_timestamp:
            if source_timestamps is None:
                raise ValueError(f"{SOURCE_TIMESTAMP} is a feature, so source_timestamps is required")
            stamps = np.asarray(source_timestamps, dtype=np.float64)
            if stamps.shape != (length,):
                raise ValueError(f"expected {length} source timestamps, got shape {stamps.shape}")
            columns[SOURCE_TIMESTAMP] = seconds_to_ns(stamps)
        self._buffer.append(columns)
        self._buffer_rows += length

        record = EpisodeRecord(
            episode_index=episode_index,
            tasks=[task],
            length=length,
            data_chunk_index=self._data_chunk_index,
            data_file_index=self._data_file_index,
            dataset_from_index=self._total_frames,
            dataset_to_index=self._total_frames + length,
            videos=dict(videos),
        )
        self._episodes.append(record)
        self._total_frames += length
        return record

    def _roll_data_file_if_needed(self) -> None:
        """Flush the buffer and advance the file index once it exceeds the budget."""
        limit = self.config.data_files_size_in_mb * 1024 * 1024
        # 2 float32 vectors + 5 int64 scalars + 1 float32 scalar per row, plus
        # the optional float64 source timestamp.
        row_bytes = 2 * 4 * self.state_dim + 5 * 8 + 4 + (8 if self.source_timestamp else 0)
        if self._buffer_rows * row_bytes < limit:
            return
        self._flush_data()
        self._advance_data_file()

    def _data_file_path(self, chunk_index: int, file_index: int) -> Path:
        return self.root / DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)

    def _advance_data_file(self) -> None:
        """Move to the next data file, carrying into the next chunk when full."""
        self._data_file_index += 1
        if self._data_file_index >= self.config.chunks_size:
            self._data_file_index = 0
            self._data_chunk_index += 1

    def _flush_data(self) -> None:
        if not self._buffer:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        columns = {
            name: np.concatenate([chunk[name] for chunk in self._buffer])
            for name in self._buffer[0]
        }
        # TacCap's file spells the list child "item" because it was written by an
        # older pyarrow; current pyarrow writes the spec-compliant "element". The
        # logical type is identical and LeRobot reads both, so we keep the default.
        vector_type = pa.list_(pa.field("item", pa.float32()), self.state_dim)
        fields = [
            pa.field("observation.state", vector_type, nullable=False),
            pa.field("action", vector_type, nullable=False),
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("index", pa.int64(), nullable=False),
            pa.field("task_index", pa.int64(), nullable=False),
        ]
        arrays = {
            "observation.state": _fixed_size_list(columns["observation.state"], vector_type),
            "action": _fixed_size_list(columns["action"], vector_type),
            "timestamp": pa.array(columns["timestamp"], type=pa.float32()),
            "frame_index": pa.array(columns["frame_index"], type=pa.int64()),
            "episode_index": pa.array(columns["episode_index"], type=pa.int64()),
            "index": pa.array(columns["index"], type=pa.int64()),
            "task_index": pa.array(columns["task_index"], type=pa.int64()),
        }
        if self.source_timestamp:
            fields.append(pa.field(SOURCE_TIMESTAMP, pa.int64(), nullable=False))
            arrays[SOURCE_TIMESTAMP] = pa.array(columns[SOURCE_TIMESTAMP], type=pa.int64())
        table = pa.table(arrays, schema=pa.schema(fields))
        path = self.root / DATA_PATH.format(
            chunk_index=self._data_chunk_index, file_index=self._data_file_index
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)

        self._buffer.clear()
        self._buffer_rows = 0

    # -- checkpointing -------------------------------------------------------

    def flush(self) -> None:
        """Write any buffered rows out, so disk and :meth:`state` agree."""
        self._flush_data()

    def state(self) -> dict[str, Any]:
        """Everything needed to carry on writing where this left off.

        Called with the buffer already flushed, so the parquet on disk and this
        state describe the same set of episodes.
        """
        return {
            "data_chunk_index": self._data_chunk_index,
            "data_file_index": self._data_file_index,
            "total_frames": self._total_frames,
            "tasks": list(self._tasks),
            "episodes": [asdict(record) for record in self._episodes],
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Adopt a :meth:`state` from an interrupted run."""
        if self._episodes or self._buffer:
            raise ValueError("restore must come before anything is written")
        self._data_chunk_index = int(state["data_chunk_index"])
        self._data_file_index = int(state["data_file_index"])
        self._total_frames = int(state["total_frames"])
        self._tasks = list(state["tasks"])
        # The file the interrupted run flushed to is complete; writing to it
        # again would replace its rows rather than add to them, losing every
        # episode the checkpoint says is already there.
        if self._data_file_path(self._data_chunk_index, self._data_file_index).exists():
            self._advance_data_file()
        self._episodes = [
            EpisodeRecord(
                **{
                    **record,
                    "videos": {
                        key: VideoPlacement(**placement)
                        for key, placement in record["videos"].items()
                    },
                }
            )
            for record in state["episodes"]
        ]

    # -- finalize ------------------------------------------------------------

    def finalize(self, stats: dict[str, dict[str, Any]]) -> None:
        self._flush_data()
        self._write_episodes()
        self._write_tasks()
        self._write_info()
        self._write_stats(stats)

    def _write_episodes(self) -> None:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows: dict[str, list[Any]] = {
            "episode_index": [],
            "tasks": [],
            "length": [],
            "data/chunk_index": [],
            "data/file_index": [],
            "dataset_from_index": [],
            "dataset_to_index": [],
        }
        for key in self.video_keys:
            for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
                rows[f"videos/{key}/{suffix}"] = []
        rows["meta/episodes/chunk_index"] = []
        rows["meta/episodes/file_index"] = []
        stat_columns = [
            f"stats/{feature}/{stat}"
            for feature in self.features
            for stat in ("min", "max", "mean", "std", "count")
        ]
        for name in stat_columns:
            rows[name] = []

        for record in self._episodes:
            rows["episode_index"].append(record.episode_index)
            rows["tasks"].append(list(record.tasks))
            rows["length"].append(record.length)
            rows["data/chunk_index"].append(record.data_chunk_index)
            rows["data/file_index"].append(record.data_file_index)
            rows["dataset_from_index"].append(record.dataset_from_index)
            rows["dataset_to_index"].append(record.dataset_to_index)
            for key in self.video_keys:
                placement = record.videos[key]
                rows[f"videos/{key}/chunk_index"].append(placement.chunk_index)
                rows[f"videos/{key}/file_index"].append(placement.file_index)
                rows[f"videos/{key}/from_timestamp"].append(placement.from_timestamp)
                rows[f"videos/{key}/to_timestamp"].append(placement.to_timestamp)
            rows["meta/episodes/chunk_index"].append(record.meta_chunk_index)
            rows["meta/episodes/file_index"].append(record.meta_file_index)
            for name in stat_columns:
                _, feature, stat = name.split("/", 2)
                try:
                    rows[name].append(record.stats[feature][stat])
                except KeyError:
                    raise ValueError(
                        f"episode {record.episode_index} has no {stat!r} for {feature!r}"
                    ) from None

        frame = pd.DataFrame(rows)
        fields = []
        for name in frame.columns:
            if name == "tasks":
                dtype: Any = pa.list_(pa.string())
            elif name.startswith("stats/"):
                dtype = self._stat_type(name)
            elif name.endswith("timestamp"):
                dtype = pa.float64()
            else:
                dtype = pa.int64()
            fields.append(pa.field(name, dtype, nullable=False))

        path = self.root / EPISODES_PATH.format(chunk_index=0, file_index=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame, schema=pa.schema(fields), preserve_index=False), path)

    def _stat_type(self, column: str):
        """Arrow type of one ``stats/<feature>/<stat>`` column.

        Matches what LeRobot's own writer produces through pandas: ``count`` is a
        one-element int list, image stats are ``(3, 1, 1)`` nested lists, every
        other stat a list with one entry per component.
        """
        import pyarrow as pa

        _, feature, stat = column.split("/", 2)
        if stat == "count":
            return pa.list_(pa.int64())
        if feature in self.video_keys:
            return pa.list_(pa.list_(pa.list_(pa.float64())))
        return pa.list_(pa.float64())

    def _write_tasks(self) -> None:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        frame = pd.DataFrame(
            {"task_index": list(range(len(self._tasks)))},
            index=pd.Index(self._tasks, name="task"),
        )
        path = self.root / "meta" / "tasks.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(frame), path)

    def _write_info(self) -> None:
        # Key order matters in that a person opens this file to answer "how big
        # is this dataset, and what robot", and `features` is hundreds of lines.
        # Same order as every TacVerse dataset: the totals first, features last.
        info = {
            "codebase_version": self.config.codebase_version,
            "robot_type": self.config.robot_type,
            "total_episodes": len(self._episodes),
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "chunks_size": self.config.chunks_size,
            "data_files_size_in_mb": self.config.data_files_size_in_mb,
            "video_files_size_in_mb": self.config.video_files_size_in_mb,
            "fps": self.config.fps,
            "splits": {"train": f"0:{len(self._episodes)}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "features": self.features,
        }
        path = self.root / "meta" / "info.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    def _write_stats(self, stats: dict[str, dict[str, Any]]) -> None:
        path = self.root / "meta" / "stats.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")


def seconds_to_ns(seconds: np.ndarray) -> np.ndarray:
    """Unix seconds (float64) -> int64 nanoseconds, rounded."""
    return np.round(np.asarray(seconds, dtype=np.float64) * 1e9).astype(np.int64)


def _fixed_size_list(values: np.ndarray, list_type):
    import pyarrow as pa

    flat = pa.array(np.ascontiguousarray(values, dtype=np.float32).reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, type=list_type)
