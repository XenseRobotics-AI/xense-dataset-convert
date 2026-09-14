"""Configuration for RLDS to LeRobot conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConverterConfig:
    """DROID-compatible defaults, with mappings kept configurable."""

    fps: int = 15
    robot_type: str = "droid"
    task: str = "Converted from RLDS"
    state_fields: tuple[str, ...] = ("joint_position", "gripper_position")
    action_field: str = "action"
    image_fields: tuple[str, ...] = (
        "exterior_image_1_left",
        "exterior_image_2_left",
    )
    action_names: tuple[str, ...] = (
        "joint_velocity_0", "joint_velocity_1", "joint_velocity_2",
        "joint_velocity_3", "joint_velocity_4", "joint_velocity_5",
        "gripper_position",
    )
    state_names: tuple[str, ...] = ()
    max_episodes: int | None = None
    use_videos: bool = True
    vcodec: str | None = None
    show_progress: bool = True
    show_progress: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not self.state_fields:
            raise ValueError("state_fields must not be empty")
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConverterConfig":
        import yaml

        values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for key in ("state_fields", "image_fields", "action_names", "state_names"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)
