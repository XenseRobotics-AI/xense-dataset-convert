"""RLDS/TFDS -> LeRobot v3, via :mod:."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..types import ConvertResult, Format
from ._support import lerobot_totals

ACCEPTED_OPTIONS = frozenset({
    "config", "split", "fps", "robot_type", "task", "state_fields",
    "action_field", "image_fields", "max_episodes", "overwrite", "vcodec", "show_progress",
    "image_writer_threads", "encoder_threads",
})


def convert(inputs: list[Path], output: Path, options: dict[str, Any]) -> ConvertResult:
    from rlds2lerobot.config import ConverterConfig
    from rlds2lerobot.converter import RldsToLeRobotConverter

    if len(inputs) != 1:
        raise ValueError("RLDS conversion expects one generated TFDS dataset directory")
    config_path = options.get("config")
    config = ConverterConfig.from_yaml(config_path) if config_path else ConverterConfig()
    overrides = {name: options[name] for name in (
        "fps", "robot_type", "task", "state_fields", "action_field", "image_fields",
        "max_episodes", "vcodec", "show_progress", "image_writer_threads", "encoder_threads",
    ) if name in options}
    for key in ("state_fields", "image_fields"):
        if key in overrides:
            overrides[key] = tuple(overrides[key])
    config = replace(config, **overrides)
    written = RldsToLeRobotConverter(
        inputs[0], output, config, split=options.get("split", "train"),
        overwrite=bool(options.get("overwrite", False)),
    ).convert()
    return ConvertResult(Format.RLDS, Format.LEROBOT, Path(written), lerobot_totals(Path(written)))
