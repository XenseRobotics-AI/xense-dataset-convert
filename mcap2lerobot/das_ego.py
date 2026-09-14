"""DAS-Ego V6 Foxglove Protobuf MCAP reader."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .config import DAS_EGO_PROFILE, ROS2_PROFILE, ConverterConfig, normalize_topic
from .reader import (
    EpisodeData,
    TimedArray,
    VideoStreamIndex,
    _MessageDecoder,
    as_bytes,
    get_message_field,
)
from .video import VideoStreamDecoder

DAS_EGO_TASK = "DAS Ego V6 sensor capture"
DAS_EGO_ROBOT_TYPE = "das_ego_v6"
DAS_EGO_FPS = 30
DAS_EGO_IMU_TOPIC = "robot0/sensor/imu"
DAS_EGO_IMU_FEATURE = "observation.imu"
SOURCE_TIMESTAMP_FEATURE = "observation.source_timestamp"
DAS_EGO_CAMERA_RE = re.compile(r"robot0/sensor/camera([0-5])/compressed$")
DAS_EGO_CAMERA_FEATURES = tuple(f"observation.images.camera{i}" for i in range(6))


def _import_protobuf() -> tuple[Any, Any, Any]:
    try:
        from mcap.reader import NonSeekingReader, make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as error:
        raise ImportError(
            "mcap and mcap-protobuf-support are required to read DAS-Ego MCAP files"
        ) from error
    return make_reader, NonSeekingReader, DecoderFactory


def make_compatible_reader(stream: Any, decoder_factories: Iterable[Any] = ()) -> Any:
    """Open an MCAP, working around chunk indexes with no message indexes.

    Foxglove's video optimization can leave chunk index entries whose
    ``message_index_offsets`` maps are all empty. mcap-python treats that as a
    searchable index and silently yields zero messages. Linear reading is both
    correct and bounded here because callers request file order.
    """
    make_reader, NonSeekingReader, _ = _import_protobuf()
    reader = make_reader(stream, decoder_factories=decoder_factories)
    summary = reader.get_summary()
    if (
        summary is not None
        and summary.chunk_indexes
        and not any(index.message_index_offsets for index in summary.chunk_indexes)
    ):
        stream.seek(0)
        return NonSeekingReader(stream, decoder_factories=decoder_factories)
    return reader


def das_ego_camera_feature(topic: str) -> str | None:
    match = DAS_EGO_CAMERA_RE.fullmatch(normalize_topic(topic))
    if match is None:
        return None
    return f"observation.images.camera{match.group(1)}"


def _looks_like_das_ego(channels: Iterable[tuple[str, str, str]]) -> bool:
    cameras: set[str] = set()
    has_imu = False
    for topic, message_encoding, schema_name in channels:
        normalized = normalize_topic(topic)
        feature = das_ego_camera_feature(normalized)
        if (
            feature is not None
            and message_encoding == "protobuf"
            and schema_name == "foxglove.CompressedImage"
        ):
            cameras.add(feature)
        if (
            normalized == DAS_EGO_IMU_TOPIC
            and message_encoding == "protobuf"
            and schema_name == "foxglove.IMUMeasurement"
        ):
            has_imu = True
    return has_imu and cameras == set(DAS_EGO_CAMERA_FEATURES)


def detect_mcap_profile(path: Path) -> str:
    """Return ``das-ego`` for its exact six-camera/IMU layout, else ``ros2``."""
    make_reader, _, _ = _import_protobuf()
    with Path(path).open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary is not None:
            channels = []
            for channel in summary.channels.values():
                schema = summary.schemas.get(channel.schema_id)
                channels.append(
                    (
                        channel.topic,
                        channel.message_encoding,
                        schema.name if schema else "",
                    )
                )
            return DAS_EGO_PROFILE if _looks_like_das_ego(channels) else ROS2_PROFILE

    channels_by_id: dict[int, tuple[str, str, str]] = {}
    with Path(path).open("rb") as stream:
        reader = make_compatible_reader(stream)
        for schema, channel, _ in reader.iter_messages(log_time_order=False):
            channels_by_id[channel.id] = (
                channel.topic,
                channel.message_encoding,
                schema.name if schema else "",
            )
            if _looks_like_das_ego(channels_by_id.values()):
                return DAS_EGO_PROFILE
    return ROS2_PROFILE


class DasEgoEpisodeReader:
    """Two-pass reader for DAS-Ego's Protobuf camera and IMU streams."""

    def __init__(
        self,
        config: ConverterConfig,
        video_decoder_factory: Callable[[str], Any] = VideoStreamDecoder,
    ):
        self.config = config
        self.video_decoder_factory = video_decoder_factory

    def read(self, path: Path) -> EpisodeData:
        _, _, DecoderFactory = _import_protobuf()
        episode = EpisodeData(source=Path(path), profile=DAS_EGO_PROFILE)
        timestamps = {feature: [] for feature in DAS_EGO_CAMERA_FEATURES}
        topics: dict[str, str] = {}
        shapes: dict[str, tuple[int, int, int]] = {}
        formats: dict[str, str] = {}
        raw_packets = dict.fromkeys(DAS_EGO_CAMERA_FEATURES, 0)
        imu: list[TimedArray] = []

        with episode.source.open("rb") as stream:
            reader = make_compatible_reader(stream)
            decode = _MessageDecoder(DecoderFactory())
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                topic = normalize_topic(channel.topic)
                feature = das_ego_camera_feature(topic)
                if feature is not None:
                    raw_packet = raw_packets[feature]
                    raw_packets[feature] += 1
                    if raw_packet == 0:
                        continue
                    timestamps[feature].append(int(message.log_time))
                    if feature not in shapes:
                        payload = decode(schema, channel, message)
                        video_format = str(get_message_field(payload, "format")).lower()
                        image = self.video_decoder_factory(video_format).decode(
                            as_bytes(get_message_field(payload, "data"))
                        )
                        topics[feature] = channel.topic
                        shapes[feature] = tuple(int(dim) for dim in image.shape)  # type: ignore[assignment]
                        formats[feature] = video_format
                    continue

                if topic == DAS_EGO_IMU_TOPIC:
                    payload = decode(schema, channel, message)
                    angular = get_message_field(payload, "angular_velocity")
                    linear = get_message_field(payload, "linear_acceleration")
                    value = np.asarray(
                        [
                            get_message_field(angular, "x"),
                            get_message_field(angular, "y"),
                            get_message_field(angular, "z"),
                            get_message_field(linear, "x"),
                            get_message_field(linear, "y"),
                            get_message_field(linear, "z"),
                        ],
                        dtype=np.float32,
                    )
                    imu.append(TimedArray(int(message.log_time), value))

        missing = [
            feature for feature in DAS_EGO_CAMERA_FEATURES if not timestamps[feature]
        ]
        if missing:
            raise ValueError(
                f"{episode.source}: missing DAS-Ego camera data: {', '.join(missing)}"
            )
        if not imu:
            raise ValueError(
                f"{episode.source}: no IMU messages on /{DAS_EGO_IMU_TOPIC}"
            )

        fps = self.config.fps or DAS_EGO_FPS
        start_ns = max(values[0] for values in timestamps.values())
        end_ns = min(values[-1] for values in timestamps.values())
        frame_count = (end_ns - start_ns) * fps // 1_000_000_000 + 1
        if frame_count <= 0:
            raise ValueError(
                f"{episode.source}: camera streams have no overlapping time range"
            )
        positions = np.arange(frame_count, dtype=np.int64)
        episode.timeline_ns = start_ns + positions * 1_000_000_000 // fps
        episode.observations[DAS_EGO_IMU_FEATURE] = imu

        for feature in DAS_EGO_CAMERA_FEATURES:
            episode.videos[feature] = VideoStreamIndex(
                feature=feature,
                topic=topics[feature],
                timestamps_ns=np.asarray(timestamps[feature], dtype=np.int64),
                shape=shapes[feature],
                format=formats[feature],
            )

        self._validate(episode)
        return episode

    def iter_frames(
        self,
        episode: EpisodeData,
        video_index: Mapping[str, np.ndarray],
    ) -> Iterator[dict[str, np.ndarray]]:
        total = len(episode.timeline_ns) if episode.timeline_ns is not None else 0
        features = list(episode.videos)
        wanted: dict[str, np.ndarray] = {}
        for feature in features:
            index = np.asarray(video_index[feature], dtype=np.int64)
            if index.shape != (total,):
                raise ValueError(
                    f"{feature}: expected one packet number per output frame ({total}), got {index.shape}"
                )
            if np.any(np.diff(index) < 0):
                raise ValueError(f"{feature}: packet numbers must be non-decreasing")
            wanted[feature] = index

        needed = {feature: set(index.tolist()) for feature, index in wanted.items()}
        last_needed = {
            feature: int(index[-1]) for feature, index in wanted.items()
        }
        decoded: dict[str, dict[int, np.ndarray]] = {
            feature: {} for feature in features
        }
        seen = dict.fromkeys(features, -1)
        decoders: dict[str, Any] = {}
        emitted = 0
        _, _, DecoderFactory = _import_protobuf()

        with episode.source.open("rb") as stream:
            reader = make_compatible_reader(stream)
            decode = _MessageDecoder(DecoderFactory())
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                feature = das_ego_camera_feature(channel.topic)
                if feature is None or feature not in needed:
                    continue
                packet = seen[feature]
                seen[feature] += 1
                if packet < 0 or packet > last_needed[feature]:
                    continue
                payload = decode(schema, channel, message)
                decoder = decoders.get(feature)
                if decoder is None:
                    decoder = self.video_decoder_factory(
                        str(get_message_field(payload, "format")).lower()
                    )
                    decoders[feature] = decoder
                image = decoder.decode(
                    as_bytes(get_message_field(payload, "data"))
                )
                # H.264 packets are stateful. Feed valid packets preceding the
                # first selected packet to initialize codec state, but retain
                # only frames referenced by the synchronization plan.
                if packet not in needed[feature]:
                    continue
                decoded[feature][packet] = image

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
                        for key in [key for key in decoded[name] if key < floor]:
                            del decoded[name][key]
                if emitted == total:
                    break

        if emitted != total:
            raise ValueError(
                f"{episode.source}: camera packets ran out while assembling frame {emitted} of {total}"
            )

    @staticmethod
    def _validate(episode: EpisodeData) -> None:
        assert episode.timeline_ns is not None
        streams: dict[str, list[int]] = {
            "timeline": episode.timeline_ns.tolist(),
            DAS_EGO_IMU_FEATURE: [
                item.timestamp_ns for item in episode.observations[DAS_EGO_IMU_FEATURE]
            ],
        }
        streams.update(
            {
                feature: video.timestamps_ns.tolist()
                for feature, video in episode.videos.items()
            }
        )
        for name, values in streams.items():
            if any(
                current <= previous for previous, current in pairwise(values)
            ):
                raise ValueError(
                    f"{episode.source}: {name} timestamps are not strictly increasing"
                )


def reader_for_profile(
    config: ConverterConfig,
    path: Path,
    video_decoder_factory: Callable[[str], Any] = VideoStreamDecoder,
) -> tuple[str, Any]:
    """Resolve auto detection and construct the matching episode reader."""
    from .reader import McapEpisodeReader

    profile = config.profile
    if profile == "auto":
        profile = detect_mcap_profile(path)
    if profile == DAS_EGO_PROFILE:
        return profile, DasEgoEpisodeReader(config, video_decoder_factory)
    return profile, McapEpisodeReader(config, video_decoder_factory)
