"""Helpers shared by the adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import Format


def single_input(inputs: list[Path], source: Format, what: str) -> Path:
    """Return the one input this source format takes, or explain the mismatch."""
    if len(inputs) != 1:
        raise ValueError(
            f"{source} takes exactly one input ({what}), got {len(inputs)}"
        )
    return inputs[0]


def lerobot_totals(dataset_root: Path) -> dict[str, Any]:
    """Episode and frame counts from a written LeRobot dataset.

    Read back from ``meta/info.json`` because no converter returns them. Missing
    or unreadable metadata yields an empty dict: this is a convenience on top of
    a conversion that already succeeded, so it must never turn one into a
    failure.
    """
    info_path = Path(dataset_root) / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {
        key: info[key]
        for key in ("total_episodes", "total_frames")
        if key in info
    }
