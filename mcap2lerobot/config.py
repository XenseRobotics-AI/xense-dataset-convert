"""Configuration for the supported MCAP topic layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASK = "Converted from MCAP"
DEFAULT_ROBOT_TYPE = "mcap"
AUTO_PROFILE = "auto"
ROS2_PROFILE = "ros2"
DAS_EGO_PROFILE = "das-ego"
DAS_UMI_PROFILE = "das-umi"
DAS_UMI_SCHEMA_VERSION = 3
DAS_UMI_SCHEMA_VERSIONS = (1, 2, 3)
SUPPORTED_PROFILES = frozenset(
    {AUTO_PROFILE, ROS2_PROFILE, DAS_EGO_PROFILE, DAS_UMI_PROFILE}
)

# Source layouts mcap2lerobot can read. "xtac" is the default: XTacData
# recordings from the XTac-UMI-G1 grippers. "lerobot_roundtrip" is the ROS2 CDR
# layout this repository's lerobot2mcap emits, kept for the round-trip path.
DIALECT_XTAC = "xtac"
DIALECT_LEROBOT_ROUNDTRIP = "lerobot_roundtrip"
DIALECTS = (DIALECT_XTAC, DIALECT_LEROBOT_ROUNDTRIP)


def normalize_topic(topic: str) -> str:
    return topic.strip("/")


@dataclass(frozen=True)
class ConverterConfig:
    """Settings for MCAP files produced by this repository's ``lerobot2mcap``.

    ``fps``, ``task`` and ``robot_type`` are overrides. Left at ``None``, the
    values lerobot2mcap embedded in the MCAP (the source dataset's
    ``meta/info.json`` and the episode's task texts) are used; only an MCAP that
    carries none falls back to the module defaults, with ``fps`` then inferred
    from the state timestamps.
    """

    dialect: str = DIALECT_XTAC
    state_topic: str = "robot/states/data"
    action_topic: str = "robot/actions/data"
    state_field: str = "observation_state"
    action_field: str = "action"
    video_topic_prefix: str = "observation/images/"
    video_topic_suffix: str = "/video"
    fps: int | None = None
    max_sync_gap_frames: float = 1.5
    task: str | None = None
    robot_type: str | None = None
    profile: str = AUTO_PROFILE
    config: Path | None = None
    check_only: bool = False
    export_action: bool | None = None
    schema_version: int = DAS_UMI_SCHEMA_VERSION
    report_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.dialect not in DIALECTS:
            raise ValueError(
                f"Unknown dialect {self.dialect!r}; expected one of {DIALECTS}"
            )
        object.__setattr__(self, "state_topic", normalize_topic(self.state_topic))
        object.__setattr__(self, "action_topic", normalize_topic(self.action_topic))
        object.__setattr__(
            self, "video_topic_prefix", normalize_topic(self.video_topic_prefix) + "/"
        )
        suffix = self.video_topic_suffix
        object.__setattr__(self, "video_topic_suffix", "/" + suffix.strip("/"))
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be greater than zero")
        if self.max_sync_gap_frames <= 0:
            raise ValueError("max_sync_gap_frames must be greater than zero")
        if self.schema_version not in DAS_UMI_SCHEMA_VERSIONS:
            raise ValueError(
                "schema_version must be one of "
                f"{DAS_UMI_SCHEMA_VERSIONS}, got {self.schema_version!r}"
            )
        profile = self.profile.lower().strip()
        if profile not in SUPPORTED_PROFILES:
            choices = ", ".join(sorted(SUPPORTED_PROFILES))
            raise ValueError(f"profile must be one of {choices}, got {self.profile!r}")
        object.__setattr__(self, "profile", profile)
        if self.config is not None:
            object.__setattr__(self, "config", Path(self.config).expanduser())
        if self.report_dir is not None:
            object.__setattr__(self, "report_dir", Path(self.report_dir).expanduser())

    def video_feature_name(self, topic: str) -> str | None:
        normalized = normalize_topic(topic)
        if not normalized.startswith(self.video_topic_prefix):
            return None
        if not normalized.endswith(self.video_topic_suffix):
            return None
        camera = normalized[
            len(self.video_topic_prefix) : -len(self.video_topic_suffix)
        ].strip("/")
        if not camera:
            return None
        return f"observation.images.{camera.replace('/', '.')}"
