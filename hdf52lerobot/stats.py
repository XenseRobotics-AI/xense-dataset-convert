"""Dataset statistics, following LeRobot's ``compute_stats`` semantics.

Numeric columns get per-component min/max/mean/std over every frame. Image
features are sampled (LeRobot's ``len ** 0.75`` heuristic), downsampled the same
way LeRobot does, normalized to [0, 1] and reduced per channel to shape (3,1,1).
Quantiles are deliberately omitted -- the reference TacCap dataset has none.
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


class StatsAccumulator:
    """Collect per-feature samples across episodes, then reduce them once."""

    def __init__(self) -> None:
        self._numeric: dict[str, list[np.ndarray]] = {}
        self._images: dict[str, list[np.ndarray]] = {}

    def add_numeric(self, key: str, values: np.ndarray) -> None:
        """Add ``(N,)`` or ``(N, D)`` rows for a numeric feature."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        self._numeric.setdefault(key, []).append(array)

    def add_images(self, key: str, frames: list[np.ndarray]) -> None:
        """Add sampled RGB frames (HWC uint8) for an image feature."""
        for frame in frames:
            chw = np.asarray(frame).transpose(2, 0, 1)
            self._images.setdefault(key, []).append(auto_downsample(chw))

    def finalize(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}

        for key, chunks in self._numeric.items():
            # Column vectors reduce to shape (1,), matrices to (D,) -- exactly the
            # shapes TacCap's stats.json carries.
            array = np.concatenate(chunks, axis=0)
            stats[key] = {
                "min": _to_list(array.min(axis=0)),
                "max": _to_list(array.max(axis=0)),
                "mean": _to_list(array.mean(axis=0)),
                "std": _to_list(array.std(axis=0)),
                "count": [len(array)],
            }

        for key, frames in self._images.items():
            per_channel = [frame.reshape(3, -1).astype(np.float64) / 255.0 for frame in frames]
            flat = np.concatenate(per_channel, axis=1)
            stats[key] = {
                "min": _to_list(flat.min(axis=1)[:, None, None]),
                "max": _to_list(flat.max(axis=1)[:, None, None]),
                "mean": _to_list(flat.mean(axis=1)[:, None, None]),
                "std": _to_list(flat.std(axis=1)[:, None, None]),
                "count": [len(frames)],
            }

        return {key: stats[key] for key in sorted(stats)}


def _to_list(value: np.ndarray) -> Any:
    return np.asarray(value).tolist()
