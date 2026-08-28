"""Raw UMI HDF5/MKV recordings -> LeRobot v3 dataset, via :mod:`hdf52lerobot`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import ConvertResult, Format
from ._support import lerobot_totals

# The scalar overrides hdf52lerobot's own CLI exposes. `config` points at a
# stream-rule YAML; the richer structures it can carry (streams, state_layout)
# belong in that file rather than in a keyword argument.
OVERRIDE_OPTIONS = frozenset(
    {
        "fps",
        "robot_type",
        "task",
        "quat_order",
        "pair",
        "max_sync_gap_frames",
        "warn_sync_gap_frames",
        "crf",
        "gop",
        "preset",
        "workers",
    }
)
ACCEPTED_OPTIONS = OVERRIDE_OPTIONS | {"config", "overwrite"}


def convert(inputs: list[Path], output: Path, options: dict[str, Any]) -> ConvertResult:
    # hdf52lerobot needs no lerobot at runtime, but keep the import lazy anyway
    # so that merely loading the facade stays free.
    from hdf52lerobot.cli import DEFAULT_CONFIG
    from hdf52lerobot.config import ConverterConfig
    from hdf52lerobot.converter import Hdf5ToLeRobotConverter

    # Same resolution order as the CLI: the given file, else the shipped one if
    # it is present, else the built-in defaults.
    config_path = options.get("config") or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    config = ConverterConfig.from_yaml(config_path) if config_path else ConverterConfig()
    config = config.with_overrides(
        **{name: options[name] for name in OVERRIDE_OPTIONS if name in options}
    )

    converter = Hdf5ToLeRobotConverter(
        inputs=inputs,
        output_root=output,
        config=config,
        overwrite=bool(options.get("overwrite", False)),
    )
    written = Path(converter.convert())

    return ConvertResult(
        source=Format.UMI_HDF5,
        target=Format.LEROBOT,
        output_path=written,
        details=lerobot_totals(written),
    )
