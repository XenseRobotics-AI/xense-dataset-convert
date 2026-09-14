"""Read one ROS2 MCAP file as one LeRobot episode, in two passes.

See ``README.md`` next to this module for the design. Pass one
(:meth:`McapEpisodeReader.read`) keeps every state and action vector and, per
camera, only the packet timestamps plus the first frame's shape. Pass two
(:meth:`McapEpisodeReader.iter_frames`) decodes exactly the packets the pairing
plan asks for, in stream order, and hands them out one state at a time, so the
decoded frames alive at any moment are a few per camera rather than the episode.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import numpy as np

from .config import ConverterConfig, normalize_topic
from .video import VideoStreamDecoder

# Attachment names lerobot2mcap uses for the source dataset's metadata. They are
# the files' paths relative to its staging directory, which is how tabular2mcap
# names attachments; matched by suffix so a prefix added by another tool is fine.
LEROBOT_INFO_ATTACHMENT = "meta/info.json"
LEROBOT_EPISODE_ATTACHMENT = "meta/episode.json"


@dataclass(frozen=True)
class TimedArray:
    timestamp_ns: int
    value: np.ndarray
    task_index: int | None = None
    """The row's ``task_index`` column, when the message carries it."""


@dataclass
class VideoStreamIndex:
    """One camera's packets as pass one saw them: when they are, not what they show."""

    feature: str
    topic: str
    """The channel topic exactly as recorded in the file."""
    timestamps_ns: np.ndarray
    """``(packets,)`` int64 log times, in the order pass two will meet them again."""
    shape: tuple[int, int, int]
    """``(height, width, channels)`` of the first decoded packet."""
    format: str
    """Codec name from the first packet, e.g. ``"h264"``."""

    def __len__(self) -> int:
        return int(self.timestamps_ns.size)


@dataclass
class EpisodeData:
    source: Path
    states: list[TimedArray] = field(default_factory=list)
    actions: list[TimedArray] = field(default_factory=list)
    videos: dict[str, VideoStreamIndex] = field(default_factory=dict)
    info: dict[str, Any] | None = None
    """The source dataset's ``meta/info.json``, when lerobot2mcap attached it.

    The ``xtac`` dialect synthesises the same shape from the MCAP metadata
    records instead of reading an attachment, so the converter's feature, fps,
    robot_type and codec resolution work unchanged for both sources."""
    episode_meta: dict[str, Any] | None = None
    """lerobot2mcap's per-episode manifest: ``episode_index``, ``length``,
    ``tasks`` and ``task_index_to_task``."""
    extra_vectors: dict[str, list[TimedArray]] = field(default_factory=dict)
    """Feature -> its own timed samples, for scalar/vector observations that ride
    on their own clock and must be nearest-neighbour paired to the states like
    the action stream (e.g. the ``xtac`` dialect's
    ``observation.tracker_confidence`` from ``/pico/pose``). Empty for the
    lerobot2mcap round-trip."""
    source_metadata: dict[str, Any] | None = None
    """Raw source metadata a dialect wants to carry past the converter, e.g. the
    ``xtac`` dialect's per-episode ``episode`` / ``xense_train_mcap`` MCAP
    metadata records used to write the extra ``meta/*.json`` files."""
    profile: str = "ros2"
    timeline_ns: np.ndarray | None = None
    """Explicit output timeline. ROS2 inputs use the state timestamps instead."""
    observations: dict[str, list[TimedArray]] = field(default_factory=dict)
    """Additional profile observations, such as DAS IMU samples."""


def get_message_field(message: Any, field_name: str) -> Any:
    if isinstance(message, Mapping):
        if field_name not in message:
            raise KeyError(field_name)
        return message[field_name]
    if not hasattr(message, field_name):
        raise KeyError(field_name)
    return getattr(message, field_name)


def optional_int_field(message: Any, field_name: str) -> int | None:
    try:
        return int(get_message_field(message, field_name))
    except (KeyError, TypeError, ValueError):
        return None


def as_vector(value: Any, field_name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(
            f"Field {field_name!r} must be one-dimensional, got {result.shape}"
        )
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


def read_lerobot_attachments(
    reader: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return ``(info, episode_meta)`` from the attachments lerobot2mcap writes.

    Either is ``None`` when the MCAP does not carry it, which is what an MCAP
    from an older lerobot2mcap or from another tool looks like.
    """
    info = episode_meta = None
    for attachment in reader.iter_attachments():
        name = str(attachment.name).replace("\\", "/")
        if name.endswith(LEROBOT_INFO_ATTACHMENT):
            info = _parse_json_attachment(attachment)
        elif name.endswith(LEROBOT_EPISODE_ATTACHMENT):
            episode_meta = _parse_json_attachment(attachment)
    return info, episode_meta


def _parse_json_attachment(attachment: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(bytes(attachment.data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            f"Attachment {attachment.name!r} is not valid JSON: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError(f"Attachment {attachment.name!r} must hold a JSON object")
    return parsed


def _import_mcap() -> tuple[Any, Any]:
    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError as error:
        raise ImportError(
            "mcap and mcap-ros2-support are required to read ROS2 MCAP files"
        ) from error
    return make_reader, DecoderFactory


class _MessageDecoder:
    """CDR-decode a message on demand, caching one decoder per channel.

    Both passes iterate the raw records and decode only what they need: pass
    one skips every video payload but the first per camera, pass two skips the
    state and action messages and the video packets no state uses.
    """

    def __init__(self, factory: Any):
        self._factory = factory
        self._by_channel: dict[int, Callable[[bytes], Any]] = {}

    def __call__(self, schema: Any, channel: Any, message: Any) -> Any:
        decoder = self._by_channel.get(channel.id)
        if decoder is None:
            decoder = self._factory.decoder_for(channel.message_encoding, schema)
            if decoder is None:
                raise ValueError(
                    f"No decoder for topic {channel.topic!r} "
                    f"(encoding {channel.message_encoding!r})"
                )
            self._by_channel[channel.id] = decoder
        return decoder(message.data)


class McapEpisodeReader:
    """Decode the ROS2 schemas and topic layout emitted by lerobot2mcap."""

    def __init__(
        self,
        config: ConverterConfig,
        video_decoder_factory: Callable[[str], Any] = VideoStreamDecoder,
    ):
        self.config = config
        self.video_decoder_factory = video_decoder_factory

    # -- pass one -------------------------------------------------------------

    def read(self, path: Path) -> EpisodeData:
        """Collect states, actions, per-camera packet timestamps and the attachments."""
        make_reader, DecoderFactory = _import_mcap()

        episode = EpisodeData(source=Path(path))
        timestamps: dict[str, list[int]] = {}
        topics: dict[str, str] = {}
        shapes: dict[str, tuple[int, int, int]] = {}
        formats: dict[str, str] = {}

        with episode.source.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            decode = _MessageDecoder(DecoderFactory())
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                topic = normalize_topic(channel.topic)
                if topic == self.config.state_topic:
                    decoded = decode(schema, channel, message)
                    episode.states.append(
                        TimedArray(
                            int(message.log_time),
                            as_vector(
                                get_message_field(decoded, self.config.state_field),
                                self.config.state_field,
                            ),
                            task_index=optional_int_field(decoded, "task_index"),
                        )
                    )
                    continue
                if topic == self.config.action_topic:
                    decoded = decode(schema, channel, message)
                    episode.actions.append(
                        TimedArray(
                            int(message.log_time),
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
                if feature_name not in timestamps:
                    # First packet of this camera: decode it once, for the frame
                    # shape build_features needs and as proof the stream decodes.
                    decoded = decode(schema, channel, message)
                    video_format = str(get_message_field(decoded, "format")).lower()
                    image = self.video_decoder_factory(video_format).decode(
                        as_bytes(get_message_field(decoded, "data"))
                    )
                    topics[feature_name] = channel.topic
                    shapes[feature_name] = tuple(int(dim) for dim in image.shape)  # type: ignore[assignment]
                    formats[feature_name] = video_format
                    timestamps[feature_name] = []
                timestamps[feature_name].append(int(message.log_time))

            episode.info, episode.episode_meta = read_lerobot_attachments(reader)

        for feature_name, log_times in timestamps.items():
            episode.videos[feature_name] = VideoStreamIndex(
                feature=feature_name,
                topic=topics[feature_name],
                timestamps_ns=np.asarray(log_times, dtype=np.int64),
                shape=shapes[feature_name],
                format=formats[feature_name],
            )

        self._validate(episode)
        return episode

    # -- pass two -------------------------------------------------------------

    def iter_frames(
        self,
        episode: EpisodeData,
        video_index: Mapping[str, np.ndarray],
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield, for every state in order, the decoded image of each camera.

        Args:
            episode: What :meth:`read` returned for this file.
            video_index: Feature -> ``(len(states),)`` packet numbers, one per
                state and non-decreasing, as :func:`converter.plan_pairing`
                computes them.

        Only packets some state uses are decoded. A decoded frame is kept until
        the state that needs it has been yielded and no later state wants it,
        so the cache holds at most the cross-camera arrival skew, a few frames.
        """
        total = len(episode.states)
        features = list(episode.videos)
        if not features:
            for _ in range(total):
                yield {}
            return

        wanted: dict[str, np.ndarray] = {}
        for feature_name in features:
            index = np.asarray(video_index[feature_name], dtype=np.int64)
            if index.shape != (total,):
                raise ValueError(
                    f"{feature_name}: expected one packet number per state ({total}), "
                    f"got shape {index.shape}"
                )
            if np.any(np.diff(index) < 0):
                raise ValueError(
                    f"{feature_name}: packet numbers must be non-decreasing"
                )
            wanted[feature_name] = index
        needed = {
            feature_name: set(index.tolist()) for feature_name, index in wanted.items()
        }

        decoded: dict[str, dict[int, np.ndarray]] = {
            feature_name: {} for feature_name in features
        }
        seen = dict.fromkeys(features, 0)
        decoders: dict[str, Any] = {}
        emitted = 0

        make_reader, DecoderFactory = _import_mcap()
        with episode.source.open("rb") as stream:
            reader = make_reader(stream, decoder_factories=[DecoderFactory()])
            decode = _MessageDecoder(DecoderFactory())
            # Same iteration order as pass one, so packet k here is packet k there.
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                feature_name = self.config.video_feature_name(
                    normalize_topic(channel.topic)
                )
                if feature_name is None or feature_name not in needed:
                    continue
                packet = seen[feature_name]
                seen[feature_name] += 1
                if packet not in needed[feature_name]:
                    continue

                payload = decode(schema, channel, message)
                decoder = decoders.get(feature_name)
                if decoder is None:
                    decoder = self.video_decoder_factory(
                        str(get_message_field(payload, "format")).lower()
                    )
                    decoders[feature_name] = decoder
                decoded[feature_name][packet] = decoder.decode(
                    as_bytes(get_message_field(payload, "data"))
                )

                while emitted < total and all(
                    int(wanted[name][emitted]) in decoded[name] for name in features
                ):
                    yield {
                        name: decoded[name][int(wanted[name][emitted])]
                        for name in features
                    }
                    emitted += 1
                    if emitted == total:
                        break
                    for name in features:
                        floor = int(wanted[name][emitted])
                        for stale in [key for key in decoded[name] if key < floor]:
                            del decoded[name][stale]
                if emitted == total:
                    break

        if emitted < total:
            raise ValueError(
                f"{episode.source}: camera packets ran out while assembling frame "
                f"{emitted} of {total}"
            )

    # -- checks ---------------------------------------------------------------

    def _validate(self, episode: EpisodeData) -> None:
        if not episode.states:
            raise ValueError(
                f"{episode.source} contains no messages on {self.config.state_topic!r}"
            )
        if not episode.actions:
            raise ValueError(
                f"{episode.source} contains no messages on {self.config.action_topic!r}"
            )
        self._validate_monotonic(
            "state", [item.timestamp_ns for item in episode.states], episode.source
        )
        self._validate_monotonic(
            "action", [item.timestamp_ns for item in episode.actions], episode.source
        )
        for feature_name, index in episode.videos.items():
            self._validate_monotonic(
                feature_name, index.timestamps_ns.tolist(), episode.source
            )

    @staticmethod
    def _validate_monotonic(name: str, timestamps: list[int], source: Path) -> None:
        if any(
            current < previous for previous, current in zip(timestamps, timestamps[1:])
        ):
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
