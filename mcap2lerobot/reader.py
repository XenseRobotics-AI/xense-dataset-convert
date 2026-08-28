"""Read one ROS2 MCAP file as one LeRobot episode."""

from __future__ import annotations

from array import array
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from .config import ConverterConfig, normalize_topic
from .video import VideoStreamDecoder


@dataclass(frozen=True)
class TimedArray:
    timestamp_ns: int
    value: np.ndarray


@dataclass(frozen=True)
class TimedImage:
    timestamp_ns: int
    value: np.ndarray


@dataclass
class EpisodeData:
    source: Path
    states: list[TimedArray] = field(default_factory=list)
    actions: list[TimedArray] = field(default_factory=list)
    videos: dict[str, list[TimedImage]] = field(default_factory=dict)


def get_message_field(message: Any, field_name: str) -> Any:
    if isinstance(message, Mapping):
        if field_name not in message:
            raise KeyError(field_name)
        return message[field_name]
    if not hasattr(message, field_name):
        raise KeyError(field_name)
    return getattr(message, field_name)


def as_vector(value: Any, field_name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"Field {field_name!r} must be one-dimensional, got {result.shape}")
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"Field {field_name!r} must be numeric, got {result.dtype}")
    return np.ascontiguousarray(result)


def as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, array):
        return value.tobytes()
    return bytes(value)


class McapEpisodeReader:
    """Decode the ROS2 schemas and topic layout emitted by lerobot2mcap."""

    def __init__(
        self,
        config: ConverterConfig,
        video_decoder_factory: Callable[[str], Any] = VideoStreamDecoder,
    ):
        self.config = config
        self.video_decoder_factory = video_decoder_factory

    def read(self, path: Path) -> EpisodeData:
        try:
            from mcap.reader import make_reader
            from mcap_ros2.decoder import DecoderFactory
        except ImportError as error:
            raise ImportError(
                "mcap and mcap-ros2-support are required to read ROS2 MCAP files"
            ) from error

        episode = EpisodeData(source=Path(path))
        video_decoders: dict[str, Any] = {}

        with Path(path).open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            for _schema, channel, message, decoded in reader.iter_decoded_messages(
                log_time_order=True
            ):
                topic = normalize_topic(channel.topic)
                if topic == self.config.state_topic:
                    episode.states.append(
                        TimedArray(
                            message.log_time,
                            as_vector(
                                get_message_field(decoded, self.config.state_field),
                                self.config.state_field,
                            ),
                        )
                    )
                    continue
                if topic == self.config.action_topic:
                    episode.actions.append(
                        TimedArray(
                            message.log_time,
                            as_vector(
                                get_message_field(decoded, self.config.action_field),
                                self.config.action_field,
                            ),
                        )
                    )
                    continue

                feature_name = self.config.video_feature_name(topic)
                if feature_name is None:
                    continue
                video_format = str(get_message_field(decoded, "format")).lower()
                packet = as_bytes(get_message_field(decoded, "data"))
                decoder = video_decoders.get(feature_name)
                if decoder is None:
                    decoder = self.video_decoder_factory(video_format)
                    video_decoders[feature_name] = decoder
                image = decoder.decode(packet)
                episode.videos.setdefault(feature_name, []).append(
                    TimedImage(message.log_time, image)
                )

        self._validate(episode)
        return episode

    def _validate(self, episode: EpisodeData) -> None:
        if not episode.states:
            raise ValueError(
                f"{episode.source} contains no messages on {self.config.state_topic!r}"
            )
        if not episode.actions:
            raise ValueError(
                f"{episode.source} contains no messages on {self.config.action_topic!r}"
            )
        self._validate_monotonic("state", episode.states, episode.source)
        self._validate_monotonic("action", episode.actions, episode.source)
        for feature_name, images in episode.videos.items():
            self._validate_monotonic(feature_name, images, episode.source)

    @staticmethod
    def _validate_monotonic(name: str, values: list[Any], source: Path) -> None:
        timestamps = [item.timestamp_ns for item in values]
        if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError(f"{source}: {name} timestamps are not monotonic")


def namespace_to_dict(value: Any) -> Any:
    """Convert decoded ROS2 namespaces into JSON-friendly structures."""
    if isinstance(value, SimpleNamespace):
        return {key: namespace_to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {key: namespace_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, array)):
        return [namespace_to_dict(item) for item in value]
    return value
