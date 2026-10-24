"""Read XTacData MCAP recordings as LeRobot episodes.

XTacData is what the XTac-UMI-G1 grippers record: one ``.train.mcap`` per
episode, with a topic layout that differs from this repository's
``lerobot2mcap`` round-trip in every way that matters, so it gets its own reader
rather than a widened :class:`~mcap2lerobot.reader.McapEpisodeReader`:

* ``/train/state`` / ``/train/action`` are **JSON**, not ROS2 CDR; the 29-value
  vector is the nested ``values`` array.
* the eight cameras are **protobuf** ``foxglove.CompressedVideo`` streams of real
  inter-frame h264 (the first frame needs two packets), decoded sequentially
  rather than one-independent-packet-at-a-time.
* there are no attachments: fps, the feature names, the task and the per-camera
  calibration live in MCAP **metadata records** (``reader.iter_metadata()``).

The reader synthesises an :attr:`~mcap2lerobot.reader.EpisodeData.info` /
``episode_meta`` in the same shape the converter already reads, so
``build_features`` / ``resolve_fps`` / ``frame_task`` and friends work unchanged.
``observation.tracker_confidence`` (the PICO drift warning) is carried on
:attr:`~mcap2lerobot.reader.EpisodeData.extra_vectors` and paired to the states
by the same nearest-neighbour plan as the action stream.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from .config import ConverterConfig, normalize_topic
from .reader import (
    EpisodeData,
    TimedArray,
    VideoStreamIndex,
    as_bytes,
    as_vector,
    get_message_field,
)

logger = logging.getLogger(__name__)

DEFAULT_XTAC_ROBOT_TYPE = "xtac_umi_g1"

# Source topic (leading slash stripped) -> LeRobot feature name. fisheye is the
# wrist camera and tactile_1/tactile_2 are the left/right pads of each gripper.
XTAC_VIDEO_TOPICS: dict[str, str] = {
    "camera/head/left/h264": "observation.images.head_left",
    "camera/head/right/h264": "observation.images.head_right",
    "camera/left/fisheye/h264": "observation.images.left_wrist",
    "camera/right/fisheye/h264": "observation.images.right_wrist",
    "camera/left/tactile_1/h264": "observation.images.left_tactile_left",
    "camera/left/tactile_2/h264": "observation.images.left_tactile_right",
    "camera/right/tactile_1/h264": "observation.images.right_tactile_left",
    "camera/right/tactile_2/h264": "observation.images.right_tactile_right",
}
# Same source topic -> its role/side, for reading xumi.camera_calibration.v1
# (keyed by the leading-slash topic) back onto the feature.
XTAC_CAMERA_CALIBRATION_TOPICS = {"/" + topic: feature for topic, feature in XTAC_VIDEO_TOPICS.items()}

XTAC_STATE_TOPIC = "train/state"
XTAC_ACTION_TOPIC = "train/action"
XTAC_POSE_TOPIC = "pico/pose"

TRACKER_CONFIDENCE_FEATURE = "observation.tracker_confidence"
TRACKER_CONFIDENCE_NAMES = ["left_tracker.confidence", "right_tracker.confidence"]

# Metadata record names carried in an XTacData MCAP.
TRAIN_METADATA = "xense_train_mcap"
EPISODE_METADATA = "episode"


def _import_mcap_reader() -> Callable[..., Any]:
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError("mcap is required to read XTacData MCAP files") from error
    return make_reader


class SequentialVideoDecoder:
    """Decode an inter-frame h264 stream packet by packet, in order.

    Unlike :class:`~mcap2lerobot.video.VideoStreamDecoder`, a packet is not
    independently decodable: feeding packet *k* may return zero frames (the
    first, priming packet) or the frame of an earlier packet. Frames come out in
    packet order with a fixed one-frame latency, so the caller assigns the *n*-th
    frame produced to the *n*-th packet fed and drains the last frame with
    :meth:`flush`.
    """

    def __init__(self, codec_name: str):
        try:
            import av
        except ImportError as error:
            raise ImportError("PyAV is required to decode XTacData video packets") from error
        self._av: Any = av
        self._codec = av.CodecContext.create(codec_name, "r")

    def feed(self, data: bytes) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for packet in self._codec.parse(bytes(data)):
            for frame in self._codec.decode(packet):
                frames.append(np.ascontiguousarray(frame.to_ndarray(format="rgb24")))
        return frames

    def flush(self) -> list[np.ndarray]:
        # Drain the decoder's one-frame pipeline latency: an empty parse yields
        # the terminating packet that releases the final buffered frame.
        frames: list[np.ndarray] = []
        for packet in self._codec.parse(b""):
            for frame in self._codec.decode(packet):
                frames.append(np.ascontiguousarray(frame.to_ndarray(format="rgb24")))
        return frames


class _XTacMessageDecoder:
    """Decode a message by its channel's encoding: JSON inline, protobuf lazily."""

    def __init__(self) -> None:
        self._protobuf_factory: Any = None
        self._protobuf_by_channel: dict[int, Callable[[bytes], Any]] = {}

    def __call__(self, schema: Any, channel: Any, message: Any) -> Any:
        encoding = channel.message_encoding
        if encoding == "json":
            return json.loads(bytes(message.data).decode("utf-8"))
        if encoding == "protobuf":
            decoder = self._protobuf_by_channel.get(channel.id)
            if decoder is None:
                if self._protobuf_factory is None:
                    try:
                        from mcap_protobuf.decoder import DecoderFactory
                    except ImportError as error:
                        raise ImportError(
                            "mcap-protobuf-support is required to read XTacData video streams"
                        ) from error
                    self._protobuf_factory = DecoderFactory()
                decoder = self._protobuf_factory.decoder_for("protobuf", schema)
                if decoder is None:
                    raise ValueError(f"No protobuf decoder for topic {channel.topic!r}")
                self._protobuf_by_channel[channel.id] = decoder
            return decoder(message.data)
        raise ValueError(f"Unsupported message encoding {encoding!r} on topic {channel.topic!r}")


def _tracker_confidence(pose: Mapping[str, Any]) -> np.ndarray | None:
    """``[left, right]`` tracker confidence from one ``/pico/pose`` message.

    Confidence is PICO's drift warning (1 good, higher = degraded); it sits in
    ``state.Motion.joints[].confidence`` keyed by serial number, so the top-level
    ``joints[]`` (which carry ``side``) map serial -> side.
    """
    side_by_serial = {
        joint.get("sn"): joint.get("side")
        for joint in pose.get("joints", [])
        if joint.get("sn") and joint.get("side")
    }
    confidence: dict[str, float] = {}
    motion = pose.get("state", {}).get("Motion", {})
    for joint in motion.get("joints", []):
        side = side_by_serial.get(joint.get("sn"))
        value = joint.get("confidence")
        if side in ("left", "right") and value is not None:
            confidence[side] = float(value)
    if "left" not in confidence or "right" not in confidence:
        return None
    return np.array([confidence["left"], confidence["right"]], dtype=np.float32)


class XTacMcapReader:
    """Read one XTacData ``.train.mcap`` as one LeRobot episode.

    Mirrors :class:`~mcap2lerobot.reader.McapEpisodeReader`'s two-pass contract
    (``read`` then ``iter_frames``) so :class:`~mcap2lerobot.converter.McapToLeRobotConverter`
    drives either reader through the same protocol.
    """

    def __init__(
        self,
        config: ConverterConfig,
        video_decoder_factory: Callable[[str], Any] = SequentialVideoDecoder,
    ):
        self.config = config
        self.video_decoder_factory = video_decoder_factory

    def video_feature(self, topic: str) -> str | None:
        return XTAC_VIDEO_TOPICS.get(normalize_topic(topic))

    # -- pass one -------------------------------------------------------------

    def read(self, path: Path) -> EpisodeData:
        make_reader = _import_mcap_reader()
        episode = EpisodeData(source=Path(path))

        timestamps: dict[str, list[int]] = {}
        topics: dict[str, str] = {}
        shapes: dict[str, tuple[int, int, int]] = {}
        formats: dict[str, str] = {}
        priming: dict[str, SequentialVideoDecoder] = {}
        confidence: list[TimedArray] = []
        pico_channel: dict[str, str] = {}
        metadata_records: dict[str, Any] = {}

        state_topic = normalize_topic(XTAC_STATE_TOPIC)
        action_topic = normalize_topic(XTAC_ACTION_TOPIC)
        pose_topic = normalize_topic(XTAC_POSE_TOPIC)

        with episode.source.open("rb") as stream:
            reader = make_reader(stream)
            decode = _XTacMessageDecoder()
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                topic = normalize_topic(channel.topic)
                if topic == state_topic:
                    decoded = decode(schema, channel, message)
                    episode.states.append(
                        TimedArray(int(message.log_time), _values(decoded), task_index=0)
                    )
                    continue
                if topic == action_topic:
                    decoded = decode(schema, channel, message)
                    episode.actions.append(TimedArray(int(message.log_time), _values(decoded)))
                    continue
                if topic == pose_topic:
                    if not pico_channel:
                        # The tracker->EE extrinsic snapshot and pose_stage ride on
                        # the channel metadata, not the messages (per the schema doc).
                        pico_channel = dict(getattr(channel, "metadata", None) or {})
                    vector = _tracker_confidence(decode(schema, channel, message))
                    if vector is not None:
                        confidence.append(TimedArray(int(message.log_time), vector))
                    continue

                feature_name = self.video_feature(channel.topic)
                if feature_name is None:
                    continue
                timestamps.setdefault(feature_name, []).append(int(message.log_time))
                topics.setdefault(feature_name, channel.topic)
                if feature_name not in shapes:
                    payload = decode(schema, channel, message)
                    video_format = str(get_message_field(payload, "format")).lower()
                    formats[feature_name] = video_format
                    decoder = priming.setdefault(feature_name, self.video_decoder_factory(video_format))
                    for frame in decoder.feed(as_bytes(get_message_field(payload, "data"))):
                        shapes[feature_name] = tuple(int(dim) for dim in frame.shape)  # type: ignore[assignment]
                        break

            for record in reader.iter_metadata():
                if record.name in (TRAIN_METADATA, EPISODE_METADATA):
                    metadata_records[record.name] = dict(record.metadata)

        for feature_name, log_times in timestamps.items():
            if feature_name not in shapes:
                raise ValueError(f"{episode.source}: could not decode a frame for {feature_name}")
            episode.videos[feature_name] = VideoStreamIndex(
                feature=feature_name,
                topic=topics[feature_name],
                timestamps_ns=np.asarray(log_times, dtype=np.int64),
                shape=shapes[feature_name],
                format=formats[feature_name],
            )

        if confidence:
            episode.extra_vectors[TRACKER_CONFIDENCE_FEATURE] = confidence
        if pico_channel:
            metadata_records["pico_channel"] = pico_channel
        episode.source_metadata = metadata_records
        self._synthesize_metadata(episode, metadata_records)
        self._validate(episode)
        return episode

    def _synthesize_metadata(self, episode: EpisodeData, records: dict[str, dict[str, str]]) -> None:
        """Build the ``info`` / ``episode_meta`` shape the converter expects."""
        train = records.get(TRAIN_METADATA, {})
        source_episode = records.get(EPISODE_METADATA, {})

        names = [name for name in (train.get("dimension_names") or "").split(",") if name]
        state_size = int(episode.states[0].value.size) if episode.states else len(names)
        vector_names = names if len(names) == state_size else None
        if names and vector_names is None:
            # Everything downstream addresses these columns by name -- the
            # viewer's replay, the pose trajectories, the per-feature filters --
            # so falling back to `state_0..state_n` is the difference between a
            # readable dataset and an opaque one. Say so rather than degrade in
            # silence.
            logger.warning(
                "%s: %d dimension_names for a %d-value state; dropping them and "
                "numbering the columns instead",
                episode.source.name,
                len(names),
                state_size,
            )

        features: dict[str, Any] = {
            "observation.state": _vector_spec(state_size, vector_names),
            "action": _vector_spec(state_size, vector_names),
        }
        if TRACKER_CONFIDENCE_FEATURE in episode.extra_vectors:
            features[TRACKER_CONFIDENCE_FEATURE] = _vector_spec(
                len(TRACKER_CONFIDENCE_NAMES), TRACKER_CONFIDENCE_NAMES
            )

        fps = _int_or_none(train.get("fps"))
        for feature_name, index in episode.videos.items():
            features[feature_name] = {
                "dtype": "video",
                "shape": list(index.shape),
                "names": ["height", "width", "channels"],
                "info": {"video.codec": index.format, "video.fps": fps},
            }

        info: dict[str, Any] = {
            "codebase_version": "v3.0",
            "robot_type": self.config.robot_type or DEFAULT_XTAC_ROBOT_TYPE,
            "features": features,
        }
        if fps:
            info["fps"] = fps
        episode.info = info

        # LeRobot's standard tasks.parquet stores task text, not separate
        # task_id/task_name/prompt columns. Use the human-readable prompt as
        # that task text; preserve the source identifiers in task_metadata.json.
        task = (
            self.config.task
            or source_episode.get("prompt")
            or source_episode.get("task_name")
            or source_episode.get("task_id")
            or None
        )
        if task:
            episode.episode_meta = {"tasks": [task], "task_index_to_task": {"0": task}}

    # -- pass two -------------------------------------------------------------

    def iter_frames(
        self,
        episode: EpisodeData,
        video_index: Mapping[str, np.ndarray],
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield the decoded image of each camera for every state, in order.

        The streams are inter-frame, so every packet is decoded in sequence; the
        1:1 state/packet pairing XTacData records makes that exactly the work
        needed. Frames are matched to packets by production order (see
        :class:`SequentialVideoDecoder`) and pruned once past.
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
                raise ValueError(f"{feature_name}: packet numbers must be non-decreasing")
            wanted[feature_name] = index
        needed = {feature_name: set(index.tolist()) for feature_name, index in wanted.items()}

        decoded: dict[str, dict[int, np.ndarray]] = {feature_name: {} for feature_name in features}
        produced = dict.fromkeys(features, 0)
        decoders: dict[str, Any] = {}
        emitted = 0

        def emit_ready() -> Iterator[dict[str, np.ndarray]]:
            nonlocal emitted
            while emitted < total and all(
                int(wanted[name][emitted]) in decoded[name] for name in features
            ):
                yield {name: decoded[name][int(wanted[name][emitted])] for name in features}
                emitted += 1
                if emitted == total:
                    return
                for name in features:
                    floor = int(wanted[name][emitted])
                    for stale in [key for key in decoded[name] if key < floor]:
                        del decoded[name][stale]

        def store(feature_name: str, frames: list[np.ndarray]) -> None:
            for frame in frames:
                packet = produced[feature_name]
                if packet in needed[feature_name]:
                    decoded[feature_name][packet] = frame
                produced[feature_name] += 1

        make_reader = _import_mcap_reader()
        with episode.source.open("rb") as stream:
            reader = make_reader(stream)
            decode = _XTacMessageDecoder()
            for schema, channel, message in reader.iter_messages(log_time_order=True):
                feature_name = self.video_feature(channel.topic)
                if feature_name is None or feature_name not in needed:
                    continue
                payload = decode(schema, channel, message)
                decoder = decoders.get(feature_name)
                if decoder is None:
                    decoder = self.video_decoder_factory(
                        str(get_message_field(payload, "format")).lower()
                    )
                    decoders[feature_name] = decoder
                store(feature_name, decoder.feed(as_bytes(get_message_field(payload, "data"))))
                if emitted < total:
                    yield from emit_ready()

            for feature_name, decoder in decoders.items():
                store(feature_name, decoder.flush())
            yield from emit_ready()

        if emitted < total:
            raise ValueError(
                f"{episode.source}: camera packets ran out while assembling frame "
                f"{emitted} of {total}"
            )

    # -- checks ---------------------------------------------------------------

    def _validate(self, episode: EpisodeData) -> None:
        if not episode.states:
            raise ValueError(f"{episode.source} contains no messages on {XTAC_STATE_TOPIC!r}")
        if not episode.actions:
            raise ValueError(f"{episode.source} contains no messages on {XTAC_ACTION_TOPIC!r}")
        _validate_monotonic("state", [item.timestamp_ns for item in episode.states], episode.source)
        _validate_monotonic("action", [item.timestamp_ns for item in episode.actions], episode.source)
        for feature_name, index in episode.videos.items():
            _validate_monotonic(feature_name, index.timestamps_ns.tolist(), episode.source)


# -- extra metadata files (written after the dataset is finalised) ------------


def write_xtac_metadata(
    output_root: Path,
    episode_metadata: list[dict[str, Any] | None],
    robot_type: str,
) -> None:
    """Write the XTac-specific ``meta/*.json`` the LeRobot writer does not produce.

    Reproduces the reference dataset's provenance files from each episode's
    ``episode`` MCAP metadata record, in episode order: ``hardware.json``,
    ``xumi_collection_devices.json``, ``taccap_extrinsics.json``,
    ``task_metadata.json`` and the per-camera ``taccap.rectify_config`` merged
    into ``meta/info.json``. Skips silently if the records are absent (e.g. a
    hand-written MCAP).
    """
    parsed = [_parse_episode_record(record) for record in episode_metadata]
    if not any(parsed):
        return
    meta = Path(output_root) / "meta"

    _write_task_metadata(meta, parsed)
    _dump_json(meta / "hardware.json", {"robot_type": robot_type, "epochs": []})
    _write_collection_devices(meta, parsed)
    _write_taccap_extrinsics(meta, parsed)
    _inject_rectify_config(meta, parsed)


def _parse_episode_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pull the fields the provenance files need out of one ``episode`` record."""
    episode = (record or {}).get(EPISODE_METADATA)
    if not episode:
        return None
    parsed: dict[str, Any] = {
        "episode_id": episode.get("episode_id"),
        "task_id": episode.get("task_id"),
        "task_name": episode.get("task_name"),
        "prompt": episode.get("prompt"),
    }
    left = episode.get("left_gripper_open_rad")
    right = episode.get("right_gripper_open_rad")
    parsed["gripper_open_rad"] = {
        "left": float(left) if left is not None else None,
        "right": float(right) if right is not None else None,
    }
    parsed["collection_devices"] = _load_json_field(episode.get("xumi.collection_devices.v1"))
    parsed["camera_calibration"] = _load_json_field(episode.get("xumi.camera_calibration.v1"))
    # The tracker->EE extrinsic and pose_stage come from the /pico/pose channel
    # metadata, captured alongside the file's metadata records.
    pico = (record or {}).get("pico_channel") or {}
    parsed["pose_stage"] = pico.get("pose_stage")
    tracker_to_ee = pico.get("tracker_to_ee")
    parsed["tracker_to_ee"] = json.loads(tracker_to_ee) if isinstance(tracker_to_ee, str) else None
    return parsed


def _write_task_metadata(meta: Path, parsed: list[dict[str, Any] | None]) -> None:
    """Persist source task identifiers and text for each converted episode."""
    _dump_json(
        meta / "task_metadata.json",
        {
            "version": 1,
            "episodes": [
                {
                    "episode_index": index,
                    "source_episode_id": (entry or {}).get("episode_id"),
                    "task_id": (entry or {}).get("task_id"),
                    "task_name": (entry or {}).get("task_name"),
                    "prompt": (entry or {}).get("prompt"),
                }
                for index, entry in enumerate(parsed)
            ],
        },
    )


def _write_collection_devices(meta: Path, parsed: list[dict[str, Any] | None]) -> None:
    episodes: list[dict[str, Any]] = []
    legacy_count = 0
    for index, entry in enumerate(parsed):
        devices = (entry or {}).get("collection_devices") or {}
        # An episode is "legacy" when its device metadata was not captured; ours
        # always is, so missing_fields is empty and legacy is False.
        legacy = bool(devices.get("missing_fields"))
        legacy_count += int(legacy)
        episodes.append(
            {
                "devices": devices,
                "episode_index": index,
                "legacy": legacy,
                "source_episode_id": (entry or {}).get("episode_id"),
            }
        )
    _dump_json(
        meta / "xumi_collection_devices.json",
        {
            "version": 1,
            "total_episodes": len(parsed),
            "snapshot_episodes": len(parsed) - legacy_count,
            "legacy_episodes": legacy_count,
            "episodes": episodes,
        },
    )


def _write_taccap_extrinsics(meta: Path, parsed: list[dict[str, Any] | None]) -> None:
    episodes: list[dict[str, Any]] = []
    for index, entry in enumerate(parsed):
        trackers = ((entry or {}).get("collection_devices") or {}).get("trackers") or {}
        bindings = {
            side: (trackers.get(side) or {}).get("serial_number") for side in ("left", "right")
        }
        tracker_to_ee = (entry or {}).get("tracker_to_ee")
        episodes.append(
            {
                "episode_index": index,
                "gripper_open_rad": (entry or {}).get("gripper_open_rad"),
                "legacy_extrinsics": tracker_to_ee is None,
                # Fixed, not derived: the reference schema carries the field,
                # but an XTacData recording has no counterpart to read it from.
                # Deriving it would mean inventing a meaning for it here.
                "legacy_world_pose": False,
                # "ee" pose_stage means the recorded poses are already the
                # end-effector; the extrinsic below is the snapshot that produced them.
                "poses_are_ee": (entry or {}).get("pose_stage") == "ee",
                "source_episode_id": (entry or {}).get("episode_id"),
                "tracker_bindings": bindings,
                "tracker_to_ee": tracker_to_ee,
                "warnings": [],
            }
        )
    _dump_json(meta / "taccap_extrinsics.json", {"episodes": episodes})


def _inject_rectify_config(meta: Path, parsed: list[dict[str, Any] | None]) -> None:
    """Merge each tactile camera's ``taccap.rectify_config`` into ``info.json``.

    ``info.json`` is dataset-global with one block per camera, so the first
    episode's calibration is used (``base_grid`` varies slightly per episode).
    """
    calibration = next(
        (entry["camera_calibration"] for entry in parsed if entry and entry.get("camera_calibration")),
        None,
    )
    info_path = meta / "info.json"
    if not calibration or not info_path.exists():
        return
    cameras = calibration.get("cameras") or {}
    info = json.loads(info_path.read_text())
    features = info.get("features") or {}
    for topic, feature_name in XTAC_CAMERA_CALIBRATION_TOPICS.items():
        camera = cameras.get(topic)
        feature = features.get(feature_name)
        if not camera or not feature or camera.get("rectify_kind") != "tactile":
            continue
        block = feature.setdefault("info", {})
        block["taccap.rectified"] = bool(camera.get("rectified", True))
        block["taccap.rectify_kind"] = camera.get("rectify_kind")
        block["taccap.rectify_config"] = camera.get("rectify_config")
    _dump_json(info_path, info)


def _load_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            # A record that will not parse is not the same thing as a record
            # that is absent, and the provenance file cannot tell them apart
            # once this returns.
            logger.warning("Ignoring an unparseable JSON metadata field: %.80s", value)
            return {}
    return value or {}


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _values(decoded: Any) -> np.ndarray:
    return as_vector(get_message_field(decoded, "values"), "values")


def _vector_spec(size: int, names: list[str] | None) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": [size],
        "names": names if names is not None else [f"state_{index}" for index in range(size)],
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_monotonic(name: str, timestamps: list[int], source: Path) -> None:
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{source}: {name} timestamps go backwards")
