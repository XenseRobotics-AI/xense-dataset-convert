"""The vocabulary the unified interface is written in.

Kept free of any import from the four converter packages: importing
``lerobot2mcap`` pulls lerobot and therefore torch, and the whole point of the
facade is that asking *what* it can convert must not cost that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Format(Enum):
    """A dataset format the facade can read from or write to.

    Construct from a configuration file by value -- ``Format("umi_hdf5")``.
    """

    LEROBOT = "lerobot"
    """A LeRobot dataset directory (v3.0 layout; v2.x is read by lerobot2mcap)."""

    MCAP = "mcap"
    """MCAP episode files."""

    ZARR = "zarr"
    """A Zarr store in the generic schema lerobot2zarr defines."""

    UMI_HDF5 = "umi_hdf5"
    """The raw per-stream HDF5/MKV recordings written by the UMI rig."""

    RLDS = "rlds"
    """A generated RLDS/TFDS dataset, such as DROID."""

    def __str__(self) -> str:
        return self.value


class ConversionError(RuntimeError):
    """A conversion was attempted and did not produce a dataset."""


class UnsupportedConversion(ValueError):
    """No converter exists for the requested pair of formats."""


@dataclass(frozen=True)
class ConvertResult:
    """What one conversion produced.

    Deliberately narrow: none of the four converters report episode or frame
    counts, and the facade does not modify them to. Adapters that can learn
    something for free after the fact put it in ``details`` -- the two writing a
    LeRobot dataset read ``meta/info.json`` for its totals -- but a key is
    present only when it was actually read, never guessed.
    """

    source: Format
    target: Format
    output_path: Path
    details: dict[str, Any] = field(default_factory=dict)
