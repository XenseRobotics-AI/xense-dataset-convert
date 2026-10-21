"""XTac training MCAPs -> LeRobot v3, via :mod:`mcap2lerobot.xtac_pack` (Rust-exporter parity)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import ConversionError, ConvertResult, Format

ACCEPTED_OPTIONS: frozenset[str] = frozenset()


def _expand(inputs: list[Path]) -> list[Path]:
    """Files as given; a directory contributes its ``*.train.mcap`` (not recursive)."""
    files: list[Path] = []
    for entry in inputs:
        files.extend(sorted(entry.glob("*.train.mcap")) if entry.is_dir() else [entry])
    return files


def convert(inputs: list[Path], output: Path, options: dict[str, Any]) -> ConvertResult:
    # Imported here: the registry must stay import-free (see registry.py).
    from mcap2lerobot.xtac_pack import TrainMcapError, pack

    files = _expand(inputs)
    if not files:
        raise ConversionError(f"no *.train.mcap under {', '.join(map(str, inputs))}")
    try:
        result = pack(files, output)
    except TrainMcapError as error:
        raise ConversionError(str(error)) from error
    return ConvertResult(
        source=Format.XTAC_TRAIN_MCAP,
        target=Format.LEROBOT,
        output_path=result.output,
        details={"total_episodes": result.episodes, "total_frames": result.frames},
    )
