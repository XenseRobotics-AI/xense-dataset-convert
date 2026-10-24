"""Configuration for the supported MCAP topic layout."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TASK = "Converted from MCAP"
DEFAULT_ROBOT_TYPE = "mcap"

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
    values embedded in the MCAP are used; for XTac, task text falls back from
    ``prompt`` to ``task_name`` to ``task_id``. Only an MCAP that carries none
    falls back to the module defaults, with ``fps`` then inferred from the state
    timestamps.
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
    image_writer_threads: int = 0
    encoder_threads: int | None = None

    def __post_init__(self) -> None:
        if self.dialect not in DIALECTS:
            raise ValueError(f"Unknown dialect {self.dialect!r}; expected one of {DIALECTS}")
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
        if self.image_writer_threads < 0:
            raise ValueError("image_writer_threads must not be negative")
        if self.encoder_threads is not None and self.encoder_threads <= 0:
            raise ValueError("encoder_threads must be greater than zero")

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
