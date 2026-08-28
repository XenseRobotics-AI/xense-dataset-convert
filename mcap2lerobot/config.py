"""Configuration for the supported MCAP topic layout."""

from __future__ import annotations

from dataclasses import dataclass


def normalize_topic(topic: str) -> str:
    return topic.strip("/")


@dataclass(frozen=True)
class ConverterConfig:
    """Settings for MCAP files produced by this repository's former exporter."""

    state_topic: str = "robot/states/data"
    action_topic: str = "robot/actions/data"
    state_field: str = "observation_state"
    action_field: str = "action"
    video_topic_prefix: str = "observation/images/"
    video_topic_suffix: str = "/video"
    fps: int | None = None
    max_sync_gap_frames: float = 1.5
    task: str = "Converted from MCAP"
    robot_type: str = "mcap"

    def __post_init__(self) -> None:
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
