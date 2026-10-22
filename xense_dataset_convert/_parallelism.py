"""Helpers for sizing conversion work to the CPUs available to this process."""

from __future__ import annotations

import os


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting affinity when present."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)
