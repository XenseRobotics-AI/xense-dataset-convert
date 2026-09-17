"""Stopping a conversion part-way, and picking it up again.

A dataset of this size takes days, so an interrupted run must not throw its
work away -- the alternative, and what used to happen, is that stopping after
an hour cost every episode converted in it.

Three states an output directory can be in, and they have to stay
distinguishable:

* ``meta/info.json`` present -- finished. A driver skips it.
* :data:`CHECKPOINT_NAME` present, ``info.json`` absent -- interrupted, and
  resumable from exactly the episodes it names.
* neither -- nothing usable; start again.

The order matters and is the whole design. ``info.json`` is what marks a
dataset complete, so a checkpoint must never write one: a partial dataset that
looks complete is worse than one that looks broken, because nothing downstream
would ever question it. Writing the checkpoint is therefore the last thing an
interrupted run does and deleting it the last thing a finished one does.
"""

from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_NAME = "meta/.checkpoint.json"
CHECKPOINT_VERSION = 1


@dataclass
class Checkpoint:
    """What an interrupted run needs to hand to the next one."""

    version: int
    config_fingerprint: str
    converted: list[str]
    skipped: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    writer: dict[str, Any]
    videos: dict[str, dict[str, int]]
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "Checkpoint | None":
        """The checkpoint under ``root``, or None when there is none to use.

        A checkpoint beside a finished ``info.json`` is stale and ignored; so
        is one this version cannot read, or one that has been truncated -- in
        every case the caller starts over rather than building on something it
        does not understand.
        """
        path = root / CHECKPOINT_NAME
        if not path.exists() or (root / "meta" / "info.json").exists():
            return None
        try:
            raw = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if raw.get("version") != CHECKPOINT_VERSION:
            return None
        try:
            return cls(**raw)
        except TypeError:
            return None

    def save(self, root: Path) -> Path:
        """Write atomically: a torn checkpoint would be worse than none."""
        path = root / CHECKPOINT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(self.__dict__, indent=2))
        os.replace(temporary, path)
        return path

    @staticmethod
    def clear(root: Path) -> None:
        (root / CHECKPOINT_NAME).unlink(missing_ok=True)


class InterruptRequest:
    """Turns the first Ctrl-C into a request to stop at the next episode.

    A conversion interrupted mid-episode leaves a half-written mp4 and no row
    describing it, so the signal is recorded rather than acted on and the loop
    finishes what it is doing. A second signal is left to the default handler:
    someone pressing it twice means now, not soon.
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)):
        self.signals = signals
        self.requested = False
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "InterruptRequest":
        for number in self.signals:
            try:
                self._previous[number] = signal.signal(number, self._handle)
            except ValueError:
                # Not the main thread: the caller keeps whatever handling it had.
                pass
        return self

    def __exit__(self, *exception: Any) -> None:
        for number, handler in self._previous.items():
            signal.signal(number, handler)
        self._previous.clear()

    def _handle(self, number: int, frame: Any) -> None:
        self.requested = True
        # Hand the next one back to Python's default, which raises.
        signal.signal(number, self._previous.get(number, signal.default_int_handler))
