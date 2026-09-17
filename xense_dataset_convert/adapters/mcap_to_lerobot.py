"""MCAP episodes -> LeRobot v3 dataset, via :mod:`mcap2lerobot`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import ConvertResult, Format
from ._support import lerobot_totals

# The fields of mcap2lerobot's own ConverterConfig, plus repo_id, which the
# converter takes separately.
CONFIG_OPTIONS = frozenset(
    {
        "dialect",
        "state_topic",
        "action_topic",
        "state_field",
        "action_field",
        "video_topic_prefix",
        "video_topic_suffix",
        "fps",
        "max_sync_gap_frames",
        "task",
        "robot_type",
        "profile",
        "config",
        "check_only",
        "export_action",
        "report_dir",
    }
)
ACCEPTED_OPTIONS = CONFIG_OPTIONS | {"repo_id"}


def convert(inputs: list[Path], output: Path, options: dict[str, Any]) -> ConvertResult:
    # Imported here, not at module level: writing the dataset needs lerobot.
    from mcap2lerobot.config import ConverterConfig
    from mcap2lerobot.converter import McapToLeRobotConverter

    output = output.resolve()

    # Only what the caller actually set, so the dataclass defaults stand.
    config = ConverterConfig(
        **{name: options[name] for name in CONFIG_OPTIONS if name in options}
    )

    converter = McapToLeRobotConverter(
        inputs=inputs,
        output_root=output,
        # Same default the CLI uses (mcap2lerobot/cli.py:58).
        repo_id=options.get("repo_id") or output.name,
        config=config,
    )
    written = Path(converter.convert())

    return ConvertResult(
        source=Format.MCAP,
        target=Format.LEROBOT,
        output_path=written,
        details=lerobot_totals(written),
    )
