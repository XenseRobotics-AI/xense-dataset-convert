"""Dataset statistics, following LeRobot's ``compute_stats`` semantics.

Numeric columns get per-component min/max/mean/std over every frame. Image
features are sampled (LeRobot's ``len ** 0.75`` heuristic), downsampled the same
way LeRobot does, normalized to [0, 1] and reduced per channel to shape (3,1,1).
Quantiles are deliberately omitted -- the reference TacCap dataset has none, and
LeRobot's ``aggregate_stats`` only carries them when every episode has them.

Two levels are produced: per-episode stats, stored as ``stats/<feature>/<stat>``
columns of ``meta/episodes`` (what ``lerobot-edit-dataset`` re-aggregates when
episodes are deleted, split or merged), and dataset-wide stats for
``meta/stats.json``, reduced over every frame rather than re-aggregated from the
episodes, so they are exact.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_num_samples(
    dataset_len: int,
    min_num_samples: int = 100,
    max_num_samples: int = 10_000,
    power: float = 0.75,
) -> int:
    """LeRobot's sample-count heuristic (``compute_stats.estimate_num_samples``)."""
    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_positions(length: int) -> list[int]:
    """Evenly spaced frame positions to sample for image statistics."""
    count = estimate_num_samples(length)
    return np.round(np.linspace(0, length - 1, count)).astype(int).tolist()


def auto_downsample(image_chw: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    """Stride-downsample a CHW image, matching LeRobot's helper."""
    _, height, width = image_chw.shape
    if max(width, height) < max_size_threshold:
        return image_chw
    factor = int(width / target_size) if width > height else int(height / target_size)
    factor = max(factor, 1)
    return image_chw[:, ::factor, ::factor]


def _as_rows(values: np.ndarray) -> np.ndarray:
    """``(N,)`` or ``(N, D)`` -> float64 ``(N, D)``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    return array


def _prepare_image(frame_hwc: np.ndarray) -> np.ndarray:
    """HWC uint8 -> the downsampled CHW array LeRobot computes stats over."""
    return auto_downsample(np.asarray(frame_hwc).transpose(2, 0, 1))


def reduce_numeric(rows: np.ndarray) -> dict[str, Any]:
    """min/max/mean/std/count of ``(N, D)`` rows, each of shape ``(D,)``.

    Column vectors therefore reduce to shape ``(1,)`` -- exactly the shapes
    TacCap's stats.json carries for the scalar columns.
    """
    return {
        "min": _to_list(rows.min(axis=0)),
        "max": _to_list(rows.max(axis=0)),
        "mean": _to_list(rows.mean(axis=0)),
        "std": _to_list(rows.std(axis=0)),
        "count": [len(rows)],
    }


def reduce_images(frames_chw: list[np.ndarray]) -> dict[str, Any]:
    """Per-channel min/max/mean/std over prepared frames, each of shape (3,1,1)."""
    per_channel = [frame.reshape(3, -1).astype(np.float64) / 255.0 for frame in frames_chw]
    flat = np.concatenate(per_channel, axis=1)
    return {
        "min": _to_list(flat.min(axis=1)[:, None, None]),
        "max": _to_list(flat.max(axis=1)[:, None, None]),
        "mean": _to_list(flat.mean(axis=1)[:, None, None]),
        "std": _to_list(flat.std(axis=1)[:, None, None]),
        "count": [len(frames_chw)],
    }


class StatsAccumulator:
    """Collect per-feature samples across episodes, then reduce them once."""

    def __init__(self) -> None:
        self._numeric: dict[str, list[np.ndarray]] = {}
        self._images: dict[str, list[np.ndarray]] = {}

    def add_numeric(self, key: str, values: np.ndarray) -> None:
        """Add ``(N,)`` or ``(N, D)`` rows for a numeric feature."""
        self._numeric.setdefault(key, []).append(_as_rows(values))

    def add_images(self, key: str, frames: list[np.ndarray]) -> None:
        """Add sampled RGB frames (HWC uint8) for an image feature."""
        for frame in frames:
            self._images.setdefault(key, []).append(_prepare_image(frame))

    def add_episode(
        self,
        numeric: dict[str, np.ndarray],
        images: dict[str, list[np.ndarray]],
    ) -> dict[str, dict[str, Any]]:
        """Add one episode's values and return that episode's own stats.

        Args:
            numeric: Feature key -> ``(N,)`` or ``(N, D)`` values of every frame.
            images: Feature key -> sampled RGB frames (HWC uint8).

        Returns:
            ``{feature: {min, max, mean, std, count}}`` for this episode alone,
            keys sorted -- the block LeRobot stores per row of ``meta/episodes``.
        """
        episode: dict[str, dict[str, Any]] = {}
        for key, values in numeric.items():
            rows = _as_rows(values)
            self._numeric.setdefault(key, []).append(rows)
            episode[key] = reduce_numeric(rows)
        for key, frames in images.items():
            prepared = [_prepare_image(frame) for frame in frames]
            self._images.setdefault(key, []).extend(prepared)
            episode[key] = reduce_images(prepared)
        return {key: episode[key] for key in sorted(episode)}

    def finalize(self) -> dict[str, dict[str, Any]]:
        """Dataset-wide stats over every frame seen, keys sorted."""
        stats: dict[str, dict[str, Any]] = {}
        for key, chunks in self._numeric.items():
            stats[key] = reduce_numeric(np.concatenate(chunks, axis=0))
        for key, frames in self._images.items():
            stats[key] = reduce_images(frames)
        return {key: stats[key] for key in sorted(stats)}


def _to_list(value: np.ndarray) -> Any:
    return np.asarray(value).tolist()
