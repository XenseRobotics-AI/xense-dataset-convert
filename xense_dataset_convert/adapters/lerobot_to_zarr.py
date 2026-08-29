"""LeRobot dataset -> Zarr, via :mod:`lerobot2zarr`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import ConvertResult, Format
from ._support import single_input

ACCEPTED_OPTIONS = frozenset(
    {"repo_id", "episodes", "feature_metadata", "source", "chunk_bytes", "limit"}
)


def convert(inputs: list[Path], output: Path, options: dict[str, Any]) -> ConvertResult:
    # Imported here, not at module level: opening a LeRobot dataset needs lerobot.
    from lerobot2zarr.zarr_converter import LeRobotZarrConverter, open_lerobot_dataset

    dataset_root = single_input(inputs, Format.LEROBOT, "the dataset root")

    # LeRobotZarrConverter is the one converter that takes an opened dataset
    # rather than a path. Callers of the facade pass a path like everywhere
    # else, so the opening happens here; open_lerobot_dataset already defaults
    # repo_id to the directory name.
    dataset = open_lerobot_dataset(
        dataset_root,
        repo_id=options.get("repo_id"),
        episodes=options.get("episodes"),
    )

    converter = LeRobotZarrConverter(
        dataset=dataset,
        output_path=output,
        feature_metadata=options.get("feature_metadata"),
        source=options.get("source"),
        **({"chunk_bytes": options["chunk_bytes"]} if "chunk_bytes" in options else {}),
    )
    written = converter.convert(limit=options.get("limit"))

    return ConvertResult(
        source=Format.LEROBOT, target=Format.ZARR, output_path=Path(written)
    )
