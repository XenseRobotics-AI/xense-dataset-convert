"""Pair every stream against the master clock and check the pairing is sane.

The recorder drives all devices from one ~20 Hz trigger but stamps each sample
with its own arrival time, so the timestamps jitter by tens of milliseconds
around a common tick. Two pairing modes:

``index``   frame i of every stream belongs to trigger i -- correct whenever the
            streams have equal length, and immune to the arrival-time jitter.
``nearest`` re-pair by real timestamps, the rule ``mcap2lerobot`` uses for
            genuinely independent topics. Used automatically for any stream
            whose length differs from the master's.

Either way the resulting pairing is validated against the real timestamps: no
sample may sit further than ``max_sync_gap_frames`` from its master tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConverterConfig
from .source import SourceEpisode


@dataclass
class StreamSync:
    """How one stream was paired to the master clock."""

    key: str
    mode: str
    indices: np.ndarray
    gaps_s: np.ndarray
    loose: int = 0
    """Frames further from the master tick than the soft limit."""

    @property
    def replaced(self) -> int:
        """Frames whose paired index differs from the identity pairing."""
        identity = np.arange(len(self.indices))
        if len(self.indices) != len(identity):
            return len(self.indices)
        return int(np.count_nonzero(self.indices != identity))

    @property
    def duplicated(self) -> int:
        return int(len(self.indices) - len(np.unique(self.indices)))

    def describe(self) -> str:
        gaps_ms = self.gaps_s * 1000.0
        loose = f" loose={self.loose}" if self.loose else ""
        return (
            f"{self.key}: mode={self.mode} gap p50={np.percentile(gaps_ms, 50):.1f}ms "
            f"p99={np.percentile(gaps_ms, 99):.1f}ms max={gaps_ms.max():.1f}ms "
            f"replaced={self.replaced} duplicated={self.duplicated}{loose}"
        )


@dataclass
class SyncReport:
    """Pairing of every stream in one episode, plus a time-axis warp summary."""

    master_key: str
    length: int
    fps: int
    streams: dict[str, StreamSync]
    warp_s: np.ndarray

    def indices_for(self, key: str) -> np.ndarray:
        return self.streams[key].indices

    def describe(self) -> list[str]:
        lines = [sync.describe() for sync in self.streams.values()]
        warp_ms = np.abs(self.warp_s) * 1000.0
        lines.append(
            f"time-axis warp vs uniform {self.fps}Hz grid: "
            f"p50={np.percentile(warp_ms, 50):.0f}ms p95={np.percentile(warp_ms, 95):.0f}ms "
            f"max={warp_ms.max():.0f}ms"
        )
        return lines


def nearest_indices(source_timestamps: np.ndarray, master_timestamps: np.ndarray) -> np.ndarray:
    """For each master tick, the index of the closest source sample."""
    if len(source_timestamps) == 1:
        return np.zeros(len(master_timestamps), dtype=np.int64)
    right = np.searchsorted(source_timestamps, master_timestamps)
    right = np.clip(right, 1, len(source_timestamps) - 1)
    left = right - 1
    take_right = np.abs(source_timestamps[right] - master_timestamps) < np.abs(
        source_timestamps[left] - master_timestamps
    )
    return np.where(take_right, right, left).astype(np.int64)


def build_sync(episode: SourceEpisode, config: ConverterConfig) -> SyncReport:
    """Pair every stream to the master clock, raising if the pairing is too loose."""
    master_timestamps = episode.timestamps_of(config.master_clock)
    length = len(master_timestamps)

    streams: dict[str, StreamSync] = {}
    violations: list[str] = []
    for key, stream in episode.all_streams().items():
        timestamps = stream.timestamps
        mode = config.pair
        if mode == "index" and len(timestamps) != length:
            mode = "nearest"
        if mode == "index":
            indices = np.arange(length, dtype=np.int64)
        else:
            indices = nearest_indices(timestamps, master_timestamps)

        gaps = np.abs(timestamps[indices] - master_timestamps)
        loose = int(np.count_nonzero(gaps > config.warn_sync_gap_s))
        streams[key] = StreamSync(
            key=key, mode=mode, indices=indices, gaps_s=gaps, loose=loose
        )
        if gaps.max() > config.max_sync_gap_s:
            over = int(np.count_nonzero(gaps > config.max_sync_gap_s))
            violations.append(
                f"{key}: {over} frame(s) off by up to {gaps.max() * 1000:.1f}ms "
                f"(limit {config.max_sync_gap_s * 1000:.1f}ms)"
            )

    if violations:
        raise ValueError(
            f"{episode.name}: streams are not synchronized with {config.master_clock!r}: "
            + "; ".join(violations)
        )

    elapsed = master_timestamps - master_timestamps[0]
    warp = elapsed - np.arange(length) / config.fps
    return SyncReport(
        master_key=config.master_clock,
        length=length,
        fps=config.fps,
        streams=streams,
        warp_s=warp,
    )
