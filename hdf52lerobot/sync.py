"""Pair every stream against the master clock and check the pairing is sane.

The recorder drives all devices from one polling loop but stamps each sample
with its own arrival time, so the timestamps jitter by tens of milliseconds
around a common iteration. Three pairing modes:

``index``   frame i of every stream belongs to iteration i -- correct whenever
            the streams have equal length, and immune to the arrival-time jitter.
``nearest`` re-pair by real timestamps, the rule ``mcap2lerobot`` uses for
            genuinely independent topics. Used automatically for any stream
            whose length differs from the master's.
``grid``    the recorder's loop is not the dataset's clock: iterations are
            paired as in ``index`` (or ``nearest`` for odd-length streams), then
            a uniform grid at ``fps`` is laid from the first to the last master
            sample and every tick takes the iteration nearest to it. Ticks that
            fall in a stall reuse the previous iteration; iterations that fall
            between ticks are dropped. This is what makes a free-running or
            stalling recorder land on LeRobot's ``frame_index / fps`` time axis.

Either way the resulting pairing is validated against the real timestamps: no
sample may sit further than ``max_sync_gap_frames`` from its tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ConverterConfig
from .source import SourceEpisode


@dataclass
class StreamSync:
    """How one stream was paired to the output ticks."""

    key: str
    mode: str
    indices: np.ndarray
    """Source sample index for every output tick."""
    gaps_s: np.ndarray
    """|source timestamp - tick time| per output tick."""
    loose: int = 0
    """Ticks further from their source sample than the soft limit."""
    source_length: int = 0

    @property
    def replaced(self) -> int:
        """Ticks whose paired index differs from the identity pairing."""
        identity = np.arange(len(self.indices))
        if len(self.indices) != len(identity):
            return len(self.indices)
        return int(np.count_nonzero(self.indices != identity))

    @property
    def duplicated(self) -> int:
        """Ticks that reuse the source sample of the previous tick."""
        return int(np.count_nonzero(np.diff(self.indices) == 0)) if len(self.indices) > 1 else 0

    @property
    def dropped(self) -> int:
        """Source samples no tick uses."""
        return int(self.source_length - np.unique(self.indices).size) if self.source_length else 0

    def describe(self) -> str:
        gaps_ms = self.gaps_s * 1000.0
        loose = f" loose={self.loose}" if self.loose else ""
        if self.mode == "grid":
            return (
                f"{self.key}: mode=grid gap p50={np.percentile(gaps_ms, 50):.1f}ms "
                f"p99={np.percentile(gaps_ms, 99):.1f}ms max={gaps_ms.max():.1f}ms "
                f"duplicated={self.duplicated} dropped={self.dropped}{loose}"
            )
        return (
            f"{self.key}: mode={self.mode} gap p50={np.percentile(gaps_ms, 50):.1f}ms "
            f"p99={np.percentile(gaps_ms, 99):.1f}ms max={gaps_ms.max():.1f}ms "
            f"replaced={self.replaced} duplicated={self.duplicated}{loose}"
        )


@dataclass
class SyncReport:
    """Pairing of every stream in one episode, plus a time-axis summary."""

    master_key: str
    length: int
    """Output frames: master samples in ``index``/``nearest`` mode, grid ticks in ``grid`` mode."""
    fps: int
    streams: dict[str, StreamSync]
    warp_s: np.ndarray
    """Per output frame, real time minus the ``frame_index / fps`` axis.

    In ``index``/``nearest`` mode this is the master's elapsed time against a
    uniform grid, i.e. how far the recorder drifted from the declared fps. In
    ``grid`` mode it is the chosen iteration's time minus its tick, bounded by
    the pairing itself.
    """
    mode: str = "index"
    tick_timestamps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Master timestamp of the iteration each output frame was taken from
    (absolute, as recorded). Written as ``observation.source_timestamp`` when
    the config asks for it."""
    source_length: int = 0
    """Master samples in the recording (equals ``length`` outside ``grid`` mode)."""

    def indices_for(self, key: str) -> np.ndarray:
        return self.streams[key].indices

    @property
    def duplicated(self) -> int:
        """Output frames that reuse the previous frame's iteration (grid mode)."""
        return self.streams[self.master_key].duplicated

    @property
    def dropped(self) -> int:
        """Recorder iterations no output frame uses (grid mode)."""
        return self.streams[self.master_key].dropped

    def describe(self) -> list[str]:
        lines = [sync.describe() for sync in self.streams.values()]
        warp_ms = np.abs(self.warp_s) * 1000.0
        if self.mode == "grid":
            lines.append(
                f"grid {self.fps}Hz: {self.length} ticks from {self.source_length} iterations, "
                f"{self.duplicated} duplicated, {self.dropped} dropped; tick offset "
                f"p50={np.percentile(warp_ms, 50):.0f}ms p95={np.percentile(warp_ms, 95):.0f}ms "
                f"max={warp_ms.max():.0f}ms"
            )
        else:
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


def uniform_grid(master_timestamps: np.ndarray, fps: int) -> np.ndarray:
    """Tick times ``t0 + i / fps`` covering the master's span, trailing remainder dropped."""
    span = float(master_timestamps[-1] - master_timestamps[0])
    count = int(np.floor(span * fps + 1e-9)) + 1
    return master_timestamps[0] + np.arange(count) / fps


def build_sync(episode: SourceEpisode, config: ConverterConfig) -> SyncReport:
    """Pair every stream to the output ticks, raising if the pairing is too loose."""
    master_timestamps = episode.timestamps_of(config.master_clock)
    source_length = len(master_timestamps)

    # Step one: every stream against the recorder's iterations (the master's samples).
    per_iteration: dict[str, tuple[str, np.ndarray]] = {}
    for key, stream in episode.all_streams().items():
        timestamps = stream.timestamps
        mode = "nearest" if config.pair == "nearest" else "index"
        if mode == "index" and len(timestamps) != source_length:
            mode = "nearest"
        if mode == "index":
            indices = np.arange(source_length, dtype=np.int64)
        else:
            indices = nearest_indices(timestamps, master_timestamps)
        per_iteration[key] = (mode, indices)

    # Step two: which iteration each output frame takes.
    if config.pair == "grid":
        ticks = uniform_grid(master_timestamps, config.fps)
        iteration = nearest_indices(master_timestamps, ticks)
        tick_timestamps = master_timestamps[iteration]
        warp = tick_timestamps - ticks
    else:
        ticks = master_timestamps
        iteration = np.arange(source_length, dtype=np.int64)
        tick_timestamps = master_timestamps
        warp = (master_timestamps - master_timestamps[0]) - np.arange(source_length) / config.fps
    length = len(ticks)

    streams: dict[str, StreamSync] = {}
    violations: list[str] = []
    for key, stream in episode.all_streams().items():
        mode, paired = per_iteration[key]
        indices = paired[iteration]
        gaps = np.abs(stream.timestamps[indices] - ticks)
        loose = int(np.count_nonzero(gaps > config.warn_sync_gap_s))
        streams[key] = StreamSync(
            key=key,
            mode="grid" if config.pair == "grid" else mode,
            indices=indices,
            gaps_s=gaps,
            loose=loose,
            source_length=len(stream.timestamps),
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

    return SyncReport(
        master_key=config.master_clock,
        length=length,
        fps=config.fps,
        streams=streams,
        warp_s=warp,
        mode=config.pair,
        tick_timestamps=np.asarray(tick_timestamps, dtype=np.float64),
        source_length=source_length,
    )
