"""Streaming reader for generated RLDS/TFDS datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def iter_episodes(root: str | Path, split: str = "train") -> Iterator[dict[str, Any]]:
    """Yield decoded episodes without loading the complete dataset.

    ``root`` is a generated TFDS directory containing ``dataset_info.json`` and
    ``features.json`` (for example ``droid/1.0.0``). TensorFlow and TFDS are
    intentionally imported only when this function is called.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as error:
        raise ImportError(
            "RLDS conversion requires the optional dependencies; install with "
            "pip install -e '.[rlds]'"
        ) from error

    path = Path(root).expanduser()
    if not (path / "dataset_info.json").exists():
        raise FileNotFoundError(f"Not a generated TFDS dataset: {path}/dataset_info.json")
    builder = tfds.builder_from_directory(str(path))
    dataset = builder.as_dataset(split=split)
    for episode in tfds.as_numpy(dataset):
        yield _normalize_episode(episode)


def episode_count(root: str | Path, split: str = "train") -> int | None:
    """Return the split size from TFDS metadata without reading TFRecords.

    ``None`` when the metadata cannot be read: the count only labels a progress
    bar, and a caller reading episodes from somewhere other than a TFDS
    directory -- a test, a generator -- still has a conversion to run.
    """
    try:
        info = json.loads((Path(root).expanduser() / "dataset_info.json").read_text())
    except (OSError, ValueError):
        return None
    for entry in info.get("splits", []):
        if entry.get("name") == split and entry.get("shardLengths") is not None:
            return sum(int(length) for length in entry["shardLengths"])
    return None


def _normalize_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Convert TFDS Dataset features into column-oriented NumPy arrays."""
    normalized = dict(episode)
    steps = normalized.get("steps")
    if isinstance(steps, dict):
        return normalized
    if steps is None:
        return normalized
    rows = list(steps)
    if not rows:
        normalized["steps"] = {}
        return normalized
    normalized["steps"] = _stack_rows(rows)
    return normalized


def _stack_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = rows[0].keys()
    result: dict[str, Any] = {}
    for key in keys:
        values = [row[key] for row in rows]
        if isinstance(values[0], dict):
            result[key] = _stack_rows(values)
        else:
            result[key] = np.asarray(values)
    return result
