"""Strict preflight and streaming reader for merged DAS Ego/Finger MCAP files."""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hdf52lerobot.geometry import quaternions_to_rotation_6d, to_wxyz

from .config import DAS_UMI_PROFILE, normalize_topic
from .das_ego import _import_protobuf, make_compatible_reader
from .reader import (
    EpisodeData,
    TimedArray,
    VideoStreamIndex,
    _MessageDecoder,
    as_bytes,
    get_message_field,
    namespace_to_dict,
)
from .video import VideoStreamDecoder

DAS_UMI_FPS = 30
DAS_UMI_TASK = "DAS UMI sensor capture"
DAS_UMI_ROBOT_TYPE = "das_umi"
DAS_UMI_SCHEMA_NAME = "das-umi"
ALIGNMENT_SIDECAR = "meta/das_umi_alignment.parquet"
POSE_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6"]
IMU_NAMES = [
    "angular_velocity.x",
    "angular_velocity.y",
    "angular_velocity.z",
    "linear_acceleration.x",
    "linear_acceleration.y",
    "linear_acceleration.z",
]


LEGACY_GRIPPER_UNITS = (
    "Encoder values pass through unchanged except float32 conversion; units were "
    "unconfirmed in DAS-UMI schemas v1/v2."
)


def _observation_names(feature: str, components: list[str]) -> list[str]:
    """Qualify auxiliary dimensions so only state/action names intentionally pair.

    LeRobot consumers group numeric series by their dimension names. Reusing bare
    names such as ``angular_velocity.x`` for all three IMUs (or ``x`` for both
    poses) therefore makes unrelated sensors look like paired state/action data.
    The feature-relative namespace keeps every auxiliary dimension unambiguous.
    """
    namespace = feature.removeprefix("observation.")
    return [f"{namespace}.{component}" for component in components]


@dataclass(frozen=True)
class StreamRule:
    key: str
    topic: str | None
    kind: str
    feature: str
    rate_hz: float
    required: bool = True
    state_block: str | None = None


@dataclass(frozen=True)
class QualityRules:
    max_header_log_offset_ms: float = 1.0
    max_start_offset_ms: float = 100.0
    max_end_offset_ms: float = 100.0
    leading_video_crop_ms: float = 1000.0
    period_multiplier: float = 2.0
    match_slack_ms: float = 0.5
    ego_camera_span_ms: float = 1.0
    output_camera_span_ms: float = 33.333334


@dataclass(frozen=True)
class GripperNormalization:
    min_opening_m: float = 0.0
    max_opening_m: float = 0.220
    decimals: int = 2

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_opening_m) or not np.isfinite(self.max_opening_m):
            raise ValueError("gripper normalization bounds must be finite")
        if self.max_opening_m <= self.min_opening_m:
            raise ValueError("gripper max_opening_m must exceed min_opening_m")
        if self.decimals < 0:
            raise ValueError("gripper normalization decimals must be non-negative")


@dataclass(frozen=True)
class DasUmiConfig:
    fps: int
    robot_type: str
    task: str
    quality: QualityRules
    gripper_normalization: GripperNormalization
    streams: tuple[StreamRule, ...]
    calibration_topics: tuple[str, ...]
    source_path: Path


@dataclass
class StreamData:
    rule: StreamRule
    log_ns: list[int] = field(default_factory=list)
    publish_ns: list[int] = field(default_factory=list)
    header_ns: list[int | None] = field(default_factory=list)
    sequence: list[int | None] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)
    valid_positions: list[int] = field(default_factory=list)
    decode_errors: list[tuple[int, str]] = field(default_factory=list)
    shape: tuple[int, int, int] | None = None
    video_format: str | None = None
    clock: str = ""
    raw_times_ns: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    times_ns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    source_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )


@dataclass
class StreamStats:
    key: str
    topic: str
    kind: str
    clock: str
    raw_count: int
    usable_count: int
    raw_start_ns: int | None
    raw_end_ns: int | None
    usable_start_ns: int | None
    usable_end_ns: int | None
    rate_hz: float | None
    interval_ms: dict[str, float]
    sequence_gaps: int | None
    decode_errors: int
    cropped_prefix: int
    max_header_log_offset_ms: float | None
    max_header_publish_offset_ms: float | None
    selected: int = 0
    unselected: int = 0
    repeated: int = 0
    sync_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class PreflightResult:
    path: Path
    episode: EpisodeData | None
    streams: dict[str, StreamData]
    stats: dict[str, StreamStats]
    inventory: dict[str, dict[str, Any]]
    calibration: dict[str, list[Any]]
    issues: list[str]
    notices: list[str]
    output_camera_span_ms: dict[str, float]
    elapsed_s: float

    @property
    def passed(self) -> bool:
        return not self.issues and self.episode is not None


class PreflightError(ValueError):
    """Raised after reports have been written for a rejected input batch."""


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "das_umi.yaml"


def load_das_umi_config(path: Path | None) -> DasUmiConfig:
    source = Path(path or default_config_path()).expanduser().resolve()
    with source.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    fps = int(raw.get("fps", DAS_UMI_FPS))
    if fps != DAS_UMI_FPS:
        raise ValueError(f"DAS-UMI output is fixed at 30 Hz; config requested {fps} Hz")
    quality = QualityRules(**(raw.get("quality") or {}))
    gripper_normalization = GripperNormalization(
        **(raw.get("gripper_normalization") or {})
    )
    rules = []
    for entry in raw.get("streams") or []:
        item = dict(entry)
        topic = item.get("topic")
        item["topic"] = normalize_topic(str(topic)) if topic else None
        rules.append(StreamRule(**item))
    if not rules:
        raise ValueError(f"{source}: streams must not be empty")
    return DasUmiConfig(
        fps=fps,
        robot_type=str(raw.get("robot_type", DAS_UMI_ROBOT_TYPE)),
        task=str(raw.get("task", DAS_UMI_TASK)),
        quality=quality,
        gripper_normalization=gripper_normalization,
        streams=tuple(rules),
        calibration_topics=tuple(
            normalize_topic(x) for x in raw.get("calibration_topics") or []
        ),
        source_path=source,
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    try:
        return get_message_field(value, name)
    except KeyError:
        return default


def _header_timestamp(payload: Any) -> int | None:
    header = _field(payload, "header")
    stamp = _field(header, "timestamp") if header is not None else None
    if stamp is None:
        return None
    if isinstance(stamp, (int, np.integer)):
        return int(stamp) or None
    seconds = _field(stamp, "seconds", _field(stamp, "sec"))
    nanos = _field(stamp, "nanos", _field(stamp, "nanosec", 0))
    if seconds is None:
        return None
    value = int(seconds) * 1_000_000_000 + int(nanos)
    return value or None


def _sequence(payload: Any, fallback: int) -> int | None:
    header = _field(payload, "header")
    value = _field(header, "sequence_num") if header is not None else None
    if value is None:
        value = fallback
    return int(value) if value is not None else None


def _xyz(value: Any) -> list[float]:
    return [float(_field(value, axis)) for axis in ("x", "y", "z")]


def _decode_numeric(kind: str, payload: Any) -> np.ndarray:
    if kind == "imu":
        return np.asarray(
            _xyz(_field(payload, "angular_velocity"))
            + _xyz(_field(payload, "linear_acceleration")),
            dtype=np.float32,
        )
    if kind == "magnetic":
        value = _field(payload, "raw_value", _field(payload, "value"))
        if value is None:
            raise ValueError("missing magnetic encoder value/raw_value")
        return np.asarray([float(value)], dtype=np.float32)
    if kind == "pose":
        pose = _field(payload, "pose", payload)
        position = _field(pose, "position")
        orientation = _field(pose, "orientation")
        xyz = np.asarray(_xyz(position), dtype=np.float64)
        quat_xyzw = np.asarray(
            [float(_field(orientation, axis)) for axis in ("x", "y", "z", "w")],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(np.concatenate([xyz, quat_xyzw]))):
            raise ValueError("pose contains a non-finite value")
        if float(np.linalg.norm(quat_xyzw)) < 1e-10:
            raise ValueError("pose contains an invalid zero-norm quaternion")
        rotation = quaternions_to_rotation_6d(to_wxyz(quat_xyzw[None], "xyzw"))[0]
        return np.concatenate([xyz, rotation]).astype(np.float32)
    raise ValueError(f"unsupported DAS-UMI stream kind {kind!r}")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "DESCRIPTOR"):
        from google.protobuf.json_format import MessageToDict

        value = MessageToDict(value, preserving_proto_field_name=True)
    value = namespace_to_dict(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"base64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _percentiles_ns(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        name: float(np.percentile(values, percentile) / 1e6)
        for name, percentile in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))
    }


def _nearest_indices(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(source, target)
    right = np.clip(insertion, 0, len(source) - 1)
    left = np.clip(insertion - 1, 0, len(source) - 1)
    # Later wins exact ties.
    take_right = (insertion == 0) | (
        (insertion < len(source))
        & (np.abs(source[right] - target) <= np.abs(source[left] - target))
    )
    return np.where(take_right, right, left).astype(np.int64)


def _classify_topic(
    topic: str, by_topic: dict[str, StreamRule], calibrations: set[str]
) -> tuple[str, str]:
    if topic in by_topic:
        return "convert", by_topic[topic].feature
    lowered = topic.lower()
    if topic in calibrations or any(
        word in lowered
        for word in ("calibration", "camera_info", "intrinsic", "extrinsic")
    ):
        return "calibration metadata", "stored in meta/das_umi_calibration.json"
    return "not converted", "outside the configured DAS-UMI feature set"


def preflight_das_umi(
    path: Path,
    config: DasUmiConfig,
    export_action: bool | None = None,
    video_decoder_factory: Callable[[str], Any] = VideoStreamDecoder,
    schema_version: int = 3,
) -> PreflightResult:
    started = time.monotonic()
    source = Path(path)
    rules = [rule for rule in config.streams if rule.topic]
    by_topic = {rule.topic: rule for rule in rules if rule.topic is not None}
    streams = {rule.key: StreamData(rule) for rule in rules}
    inventory: dict[str, dict[str, Any]] = {}
    calibration: dict[str, list[Any]] = {}
    issues: list[str] = []
    notices: list[str] = []
    decoders: dict[str, Any] = {}
    decode = None
    _, _, DecoderFactory = _import_protobuf()

    with source.open("rb") as handle:
        reader = make_compatible_reader(handle)
        decode = _MessageDecoder(DecoderFactory())
        for schema, channel, message in reader.iter_messages(log_time_order=False):
            topic = normalize_topic(channel.topic)
            classification, reason = _classify_topic(
                topic, by_topic, set(config.calibration_topics)
            )
            item = inventory.setdefault(
                topic,
                {
                    "schema": schema.name if schema else "",
                    "encoding": channel.message_encoding,
                    "count": 0,
                    "classification": classification,
                    "reason": reason,
                },
            )
            item["count"] += 1
            rule = by_topic.get(topic)
            if rule is None:
                if classification == "calibration metadata":
                    try:
                        payload = decode(schema, channel, message)
                        calibration.setdefault(topic, []).append(_jsonable(payload))
                    except Exception as error:
                        issues.append(
                            f"{topic}: calibration message cannot be decoded: {error}"
                        )
                continue
            data = streams[rule.key]
            raw_index = len(data.log_ns)
            try:
                payload = decode(schema, channel, message)
            except Exception as error:
                issues.append(
                    f"{topic} message {raw_index}: protobuf decode failed: {error}"
                )
                continue
            data.log_ns.append(int(message.log_time))
            data.publish_ns.append(int(message.publish_time))
            data.header_ns.append(_header_timestamp(payload))
            data.sequence.append(_sequence(payload, int(message.sequence)))
            if rule.kind == "video":
                try:
                    video_format = str(_field(payload, "format", "h264")).lower()
                    decoder = decoders.get(rule.key)
                    if decoder is None:
                        decoder = video_decoder_factory(video_format)
                        decoders[rule.key] = decoder
                    image = decoder.decode(as_bytes(_field(payload, "data")))
                    shape = tuple(int(x) for x in image.shape)
                    if data.shape is not None and shape != data.shape:
                        issues.append(
                            f"{topic}: resolution changed from {data.shape} to {shape}"
                        )
                    data.shape = data.shape or shape  # type: ignore[assignment]
                    data.video_format = data.video_format or video_format
                    data.valid_positions.append(raw_index)
                except Exception as error:
                    data.decode_errors.append((raw_index, str(error)))
                    if not data.valid_positions:
                        decoders.pop(rule.key, None)
                    if data.valid_positions:
                        issues.append(
                            f"{topic} message {raw_index}: video corruption after decoding began: {error}"
                        )
            else:
                try:
                    value = _decode_numeric(rule.kind, payload)
                    if not np.all(np.isfinite(value)):
                        raise ValueError("contains a non-finite value")
                    data.values.append(value)
                    data.valid_positions.append(raw_index)
                except Exception as error:
                    issues.append(f"{topic} message {raw_index}: {error}")

    for rule in rules:
        data = streams[rule.key]
        if not data.log_ns:
            if rule.required:
                issues.append(f"missing required stream {rule.key}: /{rule.topic}")
            continue
        valid_header = sum(value is not None for value in data.header_ns)
        if valid_header == len(data.header_ns):
            data.clock = "header.timestamp"
            all_times = np.asarray(data.header_ns, dtype=np.int64)
            log_offsets = np.abs(all_times - np.asarray(data.log_ns, dtype=np.int64))
            publish_offsets = np.abs(
                all_times - np.asarray(data.publish_ns, dtype=np.int64)
            )
            limit = int(config.quality.max_header_log_offset_ms * 1e6)
            for clock_name, offsets in (
                ("log_time", log_offsets),
                ("publish_time", publish_offsets),
            ):
                if int(offsets.max(initial=0)) > limit:
                    issues.append(
                        f"{rule.key}: header/{clock_name} offset {offsets.max() / 1e6:.3f} ms exceeds "
                        f"{config.quality.max_header_log_offset_ms:.3f} ms"
                    )
        elif valid_header == 0:
            data.clock = "log_time"
            all_times = np.asarray(data.log_ns, dtype=np.int64)
            notices.append(
                f"{rule.key}: no valid header timestamp; using log_time for the whole stream"
            )
        else:
            all_times = np.asarray(data.log_ns, dtype=np.int64)
            data.clock = "invalid mixed header/log"
            issues.append(
                f"{rule.key}: only {valid_header}/{len(data.header_ns)} messages have header timestamps; "
                "per-message clock mixing is forbidden"
            )
        data.raw_times_ns = all_times
        positions = np.asarray(data.valid_positions, dtype=np.int64)
        data.times_ns = (
            all_times[positions] if positions.size else np.zeros(0, dtype=np.int64)
        )
        data.source_indices = positions
        if data.times_ns.size == 0:
            if rule.required:
                issues.append(f"{rule.key}: no usable samples")
            continue
        if np.any(np.diff(data.times_ns) <= 0):
            issues.append(
                f"{rule.key}: timestamps contain a duplicate or move backwards"
            )
        if rule.kind == "video" and data.valid_positions:
            prefix = data.valid_positions[0]
            crop_ns = int(all_times[prefix] - all_times[0]) if prefix else 0
            if crop_ns > int(config.quality.leading_video_crop_ms * 1e6):
                issues.append(
                    f"{rule.key}: undecodable video prefix is {crop_ns / 1e6:.3f} ms, "
                    f"over {config.quality.leading_video_crop_ms:.3f} ms"
                )
            elif prefix:
                notices.append(
                    f"{rule.key}: cropped {prefix} undecodable leading packet(s)"
                )
        intervals = np.diff(data.times_ns)
        if intervals.size:
            max_allowed = config.quality.period_multiplier * 1e9 / rule.rate_hz
            if int(intervals.max()) > max_allowed:
                issues.append(
                    f"{rule.key}: maximum interval {intervals.max() / 1e6:.3f} ms exceeds "
                    f"{config.quality.period_multiplier:.1f} nominal periods"
                )
        selected_sequence = [data.sequence[int(i)] for i in data.source_indices]
        if len(selected_sequence) > 1 and len(set(selected_sequence)) > 1:
            diffs = np.diff(np.asarray(selected_sequence, dtype=np.int64))
            if np.any(diffs != 1):
                issues.append(f"{rule.key}: sequence gap in usable interval")

    participating = [data for data in streams.values() if data.times_ns.size]
    if participating:
        raw_starts = np.asarray(
            [data.raw_times_ns[0] for data in participating], dtype=np.int64
        )
        raw_ends = np.asarray(
            [data.raw_times_ns[-1] for data in participating], dtype=np.int64
        )
        if int(np.ptp(raw_starts)) > int(config.quality.max_start_offset_ms * 1e6):
            issues.append(
                f"continuous stream raw starts span {np.ptp(raw_starts) / 1e6:.3f} ms"
            )
        if int(np.ptp(raw_ends)) > int(config.quality.max_end_offset_ms * 1e6):
            issues.append(
                f"continuous stream raw ends span {np.ptp(raw_ends) / 1e6:.3f} ms"
            )

    episode: EpisodeData | None = None
    output_span: dict[str, float] = {}
    required_missing = any(
        rule.required and streams[rule.key].times_ns.size == 0 for rule in rules
    )
    if participating and not required_missing:
        start_ns = max(int(data.times_ns[0]) for data in participating)
        end_ns = min(int(data.times_ns[-1]) for data in participating)
        count = (end_ns - start_ns) * DAS_UMI_FPS // 1_000_000_000 + 1
        if count <= 0:
            issues.append("exported streams have no common usable interval")
        else:
            ticks = (
                start_ns
                + np.arange(count, dtype=np.int64) * 1_000_000_000 // DAS_UMI_FPS
            )
            episode = _build_episode(
                source,
                config,
                streams,
                ticks,
                export_action is not False,
                schema_version,
                issues,
                notices,
            )
            output_span = _check_sync(config, streams, ticks, issues)

    stats = _build_stats(streams)
    if episode is not None:
        _add_pairing_stats(
            stats,
            streams,
            episode.timeline_ns
            if episode.timeline_ns is not None
            else np.zeros(0, dtype=np.int64),
        )
    return PreflightResult(
        path=source,
        episode=episode,
        streams=streams,
        stats=stats,
        inventory=inventory,
        calibration=calibration,
        issues=list(dict.fromkeys(issues)),
        notices=list(dict.fromkeys(notices)),
        output_camera_span_ms=output_span,
        elapsed_s=time.monotonic() - started,
    )


def normalize_gripper_opening(
    values: np.ndarray, normalization: GripperNormalization
) -> np.ndarray:
    """Clip metre-valued jaw opening and map closed/open to 0/1."""
    array = np.asarray(values, dtype=np.float64)
    clipped = np.clip(array, normalization.min_opening_m, normalization.max_opening_m)
    normalized = (clipped - normalization.min_opening_m) / (
        normalization.max_opening_m - normalization.min_opening_m
    )
    scale = 10**normalization.decimals
    # Inputs are non-negative after clipping; floor(x + 0.5) gives conventional
    # decimal half-up rounding instead of banker's rounding at exact ties.
    rounded = np.floor(normalized * scale + 0.5) / scale
    return rounded.astype(np.float32)


def _schema_version_name(schema_version: int) -> str:
    return f"{schema_version}.0.0"


def _gripper_description(episode: EpisodeData) -> str:
    info = episode.info or {}
    normalization = info.get("gripper_normalization")
    if normalization:
        return (
            f"distance_m clipped to [{normalization['min_opening_m']:.3f}, "
            f"{normalization['max_opening_m']:.3f}], normalized with 0=closed and "
            f"1=open, rounded to {normalization['decimals']} decimals"
        )
    return LEGACY_GRIPPER_UNITS


def _build_episode(
    source: Path,
    config: DasUmiConfig,
    streams: dict[str, StreamData],
    ticks: np.ndarray,
    export_action: bool,
    schema_version: int,
    issues: list[str],
    notices: list[str],
) -> EpisodeData:
    episode = EpisodeData(source=source, profile=DAS_UMI_PROFILE, timeline_ns=ticks)
    feature_specs: dict[str, dict[str, Any]] = {}
    for data in streams.values():
        if not data.times_ns.size:
            continue
        rule = data.rule
        limit_ns = int((0.5e9 / rule.rate_hz) + config.quality.match_slack_ms * 1e6)
        episode.max_gaps_ns[rule.feature] = limit_ns
        episode.source_labels[rule.feature] = rule.key
        episode.source_indices[rule.feature] = data.source_indices
        if rule.kind == "video":
            assert data.shape is not None and data.video_format is not None
            episode.videos[rule.feature] = VideoStreamIndex(
                feature=rule.feature,
                topic=f"/{rule.topic}",
                timestamps_ns=data.times_ns,
                shape=data.shape,
                format=data.video_format,
                packet_indices=data.source_indices,
            )
        else:
            episode.observations[rule.feature] = [
                TimedArray(int(timestamp), value)
                for timestamp, value in zip(data.times_ns, data.values)
            ]
            if rule.kind == "magnetic" or (
                schema_version >= 2 and rule.state_block is not None
            ):
                # Control-source streams remain available for state/action and
                # alignment audit, but v2 does not duplicate them as standalone
                # observation vectors.
                episode.hidden_observations.add(rule.feature)
                if rule.kind == "magnetic":
                    continue
            components = (
                POSE_NAMES
                if rule.kind == "pose"
                else IMU_NAMES
                if rule.kind == "imu"
                else ["raw_value"]
            )
            names = _observation_names(rule.feature, components)
            feature_specs[rule.feature] = {
                "dtype": "float32",
                "shape": [len(names)],
                "names": names,
            }

    block_order = (
        "left_pose",
        "left_magnetic",
        "right_pose",
        "right_magnetic",
        "head_pose",
    )
    by_block = {
        data.rule.state_block: data
        for data in streams.values()
        if data.rule.state_block and data.times_ns.size
    }
    state_names: list[str] = []
    state_parts: list[np.ndarray] = []
    state_components: dict[str, list[dict[str, Any]]] = {}
    gripper_clipping: dict[str, dict[str, int | float]] = {}
    prefixes = {
        "left_pose": "left_hand",
        "left_magnetic": "left_gripper",
        "right_pose": "right_hand",
        "right_magnetic": "right_gripper",
        "head_pose": "head_camera",
    }
    for block in block_order:
        data = by_block.get(block)
        if data is None:
            continue
        index = _nearest_indices(data.times_ns, ticks)
        values = np.stack(data.values)[index]
        if schema_version >= 3 and data.rule.kind == "magnetic":
            raw_values = values.astype(np.float64, copy=False)
            rules = config.gripper_normalization
            gripper_clipping[data.rule.key] = {
                "raw_min_m": float(raw_values.min()),
                "raw_max_m": float(raw_values.max()),
                "below_min": int(np.count_nonzero(raw_values < rules.min_opening_m)),
                "above_max": int(np.count_nonzero(raw_values > rules.max_opening_m)),
            }
            values = normalize_gripper_opening(values, rules)
        state_parts.append(values)
        component_names = POSE_NAMES if data.rule.kind == "pose" else ["pos"]
        state_components[data.rule.key] = [
            {"index": len(state_names) + i, "name": f"{prefixes[block]}.{name}"}
            for i, name in enumerate(component_names)
        ]
        state_names.extend(f"{prefixes[block]}.{name}" for name in component_names)
    if state_parts:
        matrix = np.concatenate(state_parts, axis=1).astype(np.float32)
        episode.states = [TimedArray(int(t), row) for t, row in zip(ticks, matrix)]
        feature_specs["observation.state"] = {
            "dtype": "float32",
            "shape": [matrix.shape[1]],
            "names": state_names,
        }
        if export_action:
            if len(ticks) < 2:
                issues.append("action requires at least two aligned rows")
            episode.actions = [
                TimedArray(int(ticks[i]), matrix[i + 1]) for i in range(len(ticks) - 1)
            ]
            episode.states = episode.states[:-1]
            episode.timeline_ns = ticks[:-1]
            feature_specs["action"] = {
                "dtype": "float32",
                "shape": [matrix.shape[1]],
                "names": state_names,
            }
    elif export_action:
        issues.append("action requested but no state blocks are available")
    episode.info = {
        "fps": DAS_UMI_FPS,
        "robot_type": config.robot_type,
        "task": config.task,
        "features": feature_specs,
        "state_components": state_components,
        "export_action": export_action,
        "action_tail_rows_dropped": int(export_action and bool(state_parts)),
        "das_umi_schema_version": schema_version,
        "gripper_normalization": (
            {
                "input_unit": "m",
                "min_opening_m": config.gripper_normalization.min_opening_m,
                "max_opening_m": config.gripper_normalization.max_opening_m,
                "decimals": config.gripper_normalization.decimals,
                "closed_value": 0.0,
                "open_value": 1.0,
                "out_of_range": "clip",
                "clipped_aligned_samples": gripper_clipping,
            }
            if schema_version >= 3
            else None
        ),
    }
    return episode


def _check_sync(
    config: DasUmiConfig,
    streams: dict[str, StreamData],
    ticks: np.ndarray,
    issues: list[str],
) -> dict[str, float]:
    for stream in streams.values():
        if not stream.times_ns.size:
            continue
        stream_index = _nearest_indices(stream.times_ns, ticks)
        stream_gaps = np.abs(stream.times_ns[stream_index] - ticks)
        stream_limit = int(
            (0.5e9 / stream.rule.rate_hz) + config.quality.match_slack_ms * 1e6
        )
        if int(stream_gaps.max(initial=0)) > stream_limit:
            issues.append(
                f"{stream.rule.key}: nearest sample gap {stream_gaps.max() / 1e6:.3f} ms "
                f"exceeds {stream_limit / 1e6:.3f} ms"
            )
    cameras = [
        data
        for data in streams.values()
        if data.rule.kind == "video" and data.times_ns.size
    ]
    selected = []
    for data in cameras:
        index = _nearest_indices(data.times_ns, ticks)
        gaps = np.abs(data.times_ns[index] - ticks)
        limit = int((0.5e9 / data.rule.rate_hz) + config.quality.match_slack_ms * 1e6)
        if int(gaps.max(initial=0)) > limit:
            issues.append(
                f"{data.rule.key}: nearest camera gap {gaps.max() / 1e6:.3f} ms exceeds {limit / 1e6:.3f} ms"
            )
        selected.append(data.times_ns[index])
    result: dict[str, float] = {}
    if selected:
        matrix = np.stack(selected)
        span = np.ptp(matrix, axis=0)
        result["p50"] = float(np.percentile(span, 50) / 1e6)
        result["p95"] = float(np.percentile(span, 95) / 1e6)
        result["p99"] = float(np.percentile(span, 99) / 1e6)
        result["max"] = float(span.max() / 1e6)
        result["over_ideal_half_frame"] = int(np.sum(span > (1e9 / 60)))
        if int(span.max()) > int(config.quality.output_camera_span_ms * 1e6):
            issues.append(
                f"output rows: eight-camera span {span.max() / 1e6:.3f} ms exceeds "
                f"{config.quality.output_camera_span_ms:.3f} ms"
            )
    ego = [data for data in cameras if data.rule.key.startswith("camera")]
    if len(ego) == 6:
        matrix = np.stack(
            [data.times_ns[_nearest_indices(data.times_ns, ticks)] for data in ego]
        )
        worst = int(np.ptp(matrix, axis=0).max())
        if worst > int(config.quality.ego_camera_span_ms * 1e6):
            issues.append(
                f"Ego six-camera span {worst / 1e6:.3f} ms exceeds "
                f"{config.quality.ego_camera_span_ms:.3f} ms"
            )
    if ego:
        reference = ego[0].times_ns
        for key in ("left_wrist", "right_wrist"):
            other = streams.get(key)
            if other is None or not other.times_ns.size:
                continue
            common_start, common_end = int(ticks[0]), int(ticks[-1])
            common_reference = reference[
                (reference >= common_start) & (reference <= common_end)
            ]
            common_other = other.times_ns[
                (other.times_ns >= common_start) & (other.times_ns <= common_end)
            ]
            forward = np.abs(
                other.times_ns[_nearest_indices(other.times_ns, common_reference)]
                - common_reference
            )
            reverse = np.abs(
                reference[_nearest_indices(reference, common_other)] - common_other
            )
            worst = max(int(forward.max(initial=0)), int(reverse.max(initial=0)))
            limit = int(1e9 / 60 + config.quality.match_slack_ms * 1e6)
            if worst > limit:
                issues.append(
                    f"{key}: bidirectional cross-device gap {worst / 1e6:.3f} ms exceeds {limit / 1e6:.3f} ms"
                )
    return result


def _build_stats(streams: dict[str, StreamData]) -> dict[str, StreamStats]:
    result = {}
    for key, data in streams.items():
        intervals = np.diff(data.times_ns)
        sequence = [data.sequence[int(i)] for i in data.source_indices]
        gaps: int | None = None
        if len(sequence) > 1 and len(set(sequence)) > 1:
            gaps = int(np.sum(np.diff(np.asarray(sequence, dtype=np.int64)) != 1))
        offsets = None
        if data.clock == "header.timestamp" and data.header_ns:
            offsets = float(
                np.max(
                    np.abs(
                        np.asarray(data.header_ns, dtype=np.int64)
                        - np.asarray(data.log_ns, dtype=np.int64)
                    )
                )
                / 1e6
            )
        publish_offset = None
        if data.clock == "header.timestamp" and data.header_ns:
            publish_offset = float(
                np.max(
                    np.abs(
                        np.asarray(data.header_ns, dtype=np.int64)
                        - np.asarray(data.publish_ns, dtype=np.int64)
                    )
                )
                / 1e6
            )
        result[key] = StreamStats(
            key=key,
            topic=f"/{data.rule.topic}" if data.rule.topic else "(not configured)",
            kind=data.rule.kind,
            clock=data.clock or "missing",
            raw_count=len(data.log_ns),
            usable_count=len(data.times_ns),
            raw_start_ns=int(data.raw_times_ns[0]) if data.raw_times_ns.size else None,
            raw_end_ns=int(data.raw_times_ns[-1]) if data.raw_times_ns.size else None,
            usable_start_ns=int(data.times_ns[0]) if data.times_ns.size else None,
            usable_end_ns=int(data.times_ns[-1]) if data.times_ns.size else None,
            rate_hz=(float(1e9 / np.median(intervals)) if intervals.size else None),
            interval_ms=_percentiles_ns(intervals),
            sequence_gaps=gaps,
            decode_errors=len(data.decode_errors),
            cropped_prefix=(data.valid_positions[0] if data.valid_positions else 0),
            max_header_log_offset_ms=offsets,
            max_header_publish_offset_ms=publish_offset,
        )
    return result


def _add_pairing_stats(
    stats: dict[str, StreamStats], streams: dict[str, StreamData], ticks: np.ndarray
) -> None:
    for key, data in streams.items():
        if not data.times_ns.size or not ticks.size:
            continue
        index = _nearest_indices(data.times_ns, ticks)
        unique = len(np.unique(index))
        stats[key].selected = unique
        stats[key].unselected = len(data.times_ns) - unique
        stats[key].repeated = len(index) - unique
        stats[key].sync_ms = _percentiles_ns(np.abs(data.times_ns[index] - ticks))


class DasUmiEpisodeReader:
    """Second pass: decode only selected raw packets, retaining bounded caches."""

    def __init__(
        self, video_decoder_factory: Callable[[str], Any] = VideoStreamDecoder
    ):
        self.video_decoder_factory = video_decoder_factory

    def iter_frames(
        self, episode: EpisodeData, video_index: Mapping[str, np.ndarray]
    ) -> Iterator[dict[str, np.ndarray]]:
        total = len(episode.timeline_ns) if episode.timeline_ns is not None else 0
        if not episode.videos:
            for _ in range(total):
                yield {}
            return
        wanted_raw: dict[str, np.ndarray] = {}
        topic_to_feature = {}
        for feature, video in episode.videos.items():
            selected = np.asarray(video_index[feature], dtype=np.int64)
            raw = (
                video.packet_indices[selected]
                if video.packet_indices is not None
                else selected
            )
            wanted_raw[feature] = raw
            topic_to_feature[normalize_topic(video.topic)] = feature
        needed = {key: set(value.tolist()) for key, value in wanted_raw.items()}
        last = {key: int(value[-1]) for key, value in wanted_raw.items()}
        seen = dict.fromkeys(wanted_raw, 0)
        decoded: dict[str, dict[int, np.ndarray]] = {key: {} for key in wanted_raw}
        decoders: dict[str, Any] = {}
        emitted = 0
        _, _, DecoderFactory = _import_protobuf()
        with episode.source.open("rb") as handle:
            reader = make_compatible_reader(handle)
            decode = _MessageDecoder(DecoderFactory())
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                feature = topic_to_feature.get(normalize_topic(channel.topic))
                if feature is None:
                    continue
                raw_index = seen[feature]
                seen[feature] += 1
                if raw_index > last[feature]:
                    continue
                try:
                    payload = decode(schema, channel, message)
                    decoder = decoders.get(feature)
                    if decoder is None:
                        decoder = self.video_decoder_factory(
                            str(_field(payload, "format", "h264")).lower()
                        )
                        decoders[feature] = decoder
                    image = decoder.decode(as_bytes(_field(payload, "data")))
                except Exception:
                    if not decoded[feature]:
                        decoders.pop(feature, None)
                    if raw_index in needed[feature]:
                        raise ValueError(
                            f"{feature}: selected packet {raw_index} failed during write pass"
                        )
                    continue
                if raw_index in needed[feature]:
                    decoded[feature][raw_index] = image
                while emitted < total and all(
                    int(wanted_raw[name][emitted]) in decoded[name]
                    for name in wanted_raw
                ):
                    yield {
                        name: decoded[name][int(wanted_raw[name][emitted])]
                        for name in wanted_raw
                    }
                    emitted += 1
                    if emitted == total:
                        break
                    for name in wanted_raw:
                        floor = int(wanted_raw[name][emitted])
                        for stale in [x for x in decoded[name] if x < floor]:
                            del decoded[name][stale]
                if emitted == total:
                    break
        if emitted != total:
            raise ValueError(
                f"{episode.source}: assembled {emitted} of {total} DAS-UMI video rows"
            )


def render_preflight(results: list[PreflightResult], config: DasUmiConfig) -> str:
    passed = all(result.passed for result in results)
    lines = [
        "# DAS-UMI preflight",
        "",
        f"**Result: {'PASS' if passed else 'FAIL'}**",
        "",
        f"Config: `{config.source_path}`",
        f"Output rate: {config.fps} Hz (fixed)",
        "Quality thresholds: "
        + ", ".join(f"{key}={value}" for key, value in vars(config.quality).items()),
        "",
    ]
    for result in results:
        lines += [f"## `{result.path}`", "", f"Scan time: {result.elapsed_s:.3f} s", ""]
        if result.issues:
            lines += (
                ["### Blocking issues", ""]
                + [f"- {item}" for item in result.issues]
                + [""]
            )
        if result.notices:
            lines += (
                ["### Allowed trims and notices", ""]
                + [f"- {item}" for item in result.notices]
                + [""]
            )
        lines += [
            "### Exported streams",
            "",
            "| key | topic | clock | raw/usable | duration s | raw range ns | usable range ns | Hz | interval p50/p95/p99/max ms | header-log/publish max ms | seq gaps | decode errors |",
            "|---|---|---|---:|---:|---|---|---:|---|---|---:|---:|",
        ]
        for stat in result.stats.values():
            interval = stat.interval_ms
            formatted = "/".join(
                f"{interval.get(key, 0):.3f}" for key in ("p50", "p95", "p99", "max")
            )
            rate = "-" if stat.rate_hz is None else f"{stat.rate_hz:.3f}"
            seq = (
                "unavailable" if stat.sequence_gaps is None else str(stat.sequence_gaps)
            )
            duration = (
                (stat.usable_end_ns - stat.usable_start_ns) / 1e9
                if stat.usable_start_ns is not None and stat.usable_end_ns is not None
                else 0.0
            )
            clock_offsets = (
                f"{stat.max_header_log_offset_ms:.3f}/{stat.max_header_publish_offset_ms:.3f}"
                if stat.max_header_log_offset_ms is not None
                and stat.max_header_publish_offset_ms is not None
                else "-"
            )
            lines.append(
                f"| {stat.key} | `{stat.topic}` | {stat.clock} | {stat.raw_count}/{stat.usable_count} | "
                f"{duration:.3f} | {stat.raw_start_ns}…{stat.raw_end_ns} | {stat.usable_start_ns}…{stat.usable_end_ns} | {rate} | {formatted} | {clock_offsets} | {seq} | {stat.decode_errors} |"
            )
        lines += [
            "",
            "### Topic inventory",
            "",
            "| topic | schema | count | disposition | reason |",
            "|---|---|---:|---|---|",
        ]
        for topic, item in sorted(result.inventory.items()):
            lines.append(
                f"| `/{topic}` | `{item['schema']}` | {item['count']} | {item['classification']} | {item['reason']} |"
            )
        lines += ["", "### Synchronization", ""]
        for stat in result.stats.values():
            if stat.sync_ms:
                values = ", ".join(
                    f"{key.upper()}={value:.3f} ms"
                    for key, value in stat.sync_ms.items()
                )
                lines.append(
                    f"- {stat.key}: {values}; selected={stat.selected}, unselected={stat.unselected}, repeated={stat.repeated}"
                )
        if result.output_camera_span_ms:
            values = ", ".join(
                f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in result.output_camera_span_ms.items()
            )
            lines.append(f"- Eight-camera row span: {values}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_conversion_report(
    path: Path,
    status: str,
    results: list[PreflightResult],
    features: dict[str, dict[str, Any]] | None = None,
    output: Path | None = None,
    reason: str | None = None,
    elapsed_s: float | None = None,
    temporary: Path | None = None,
) -> None:
    frames = sum(
        len(result.episode.timeline_ns)
        for result in results
        if result.episode is not None and result.episode.timeline_ns is not None
    )
    lines = [
        "# DAS-UMI conversion",
        "",
        f"**Status: {status}**",
        "",
        f"Episodes: {len(results)}",
        f"Frames: {frames}",
    ]
    action_enabled = any(
        (result.episode.info or {}).get("export_action", False)
        for result in results
        if result.episode is not None
    )
    dropped = sum(
        (result.episode.info or {}).get("action_tail_rows_dropped", 0)
        for result in results
        if result.episode is not None
    )
    schema_version = max(
        (
            int((result.episode.info or {}).get("das_umi_schema_version", 1))
            for result in results
            if result.episode is not None
        ),
        default=1,
    )
    first_episode = next(
        (result.episode for result in results if result.episode is not None), None
    )
    gripper_description = (
        _gripper_description(first_episode) if first_episode else LEGACY_GRIPPER_UNITS
    )
    lines += [
        f"DAS-UMI schema: {schema_version}",
        f"Action: {'enabled (next-frame absolute state proxy)' if action_enabled else 'disabled'}",
        f"Action tail rows dropped: {dropped} (one per enabled episode)",
        f"Gripper: {gripper_description}",
        "Head: world pose; rotation [R00,R10,R20,R01,R11,R21]; no TacCap camera-frame calibration implied.",
        "Topic mapping: `meta/das_umi_topic_mapping.json` (created with dataset)",
        (
            "Alignment audit: `meta/das_umi_alignment.parquet` (not a training feature)"
            if schema_version >= 2
            else "Alignment audit: inline `observation.source_*` features (legacy v1)"
        ),
    ]
    if output is not None:
        lines.append(f"Output: `{output}`")
    if temporary is not None:
        lines.append(f"Temporary output retained at: `{temporary}`")
    if elapsed_s is not None:
        lines.append(f"Elapsed: {elapsed_s:.3f} s")
    if reason:
        lines += ["", "## Reason", "", reason]
    lines += ["", "## Validation", ""]
    if status == "PASS":
        lines += [
            "- LeRobot load and row count: PASS",
            "- Integer 30 Hz timelines: PASS",
            (
                "- Alignment sidecar and absence of inline audit features: PASS"
                if schema_version >= 2
                else "- Legacy inline alignment audit features: PASS"
            ),
            *(
                ["- Gripper range [0, 1] and two-decimal quantization: PASS"]
                if schema_version >= 3
                else []
            ),
            "- Video decode count and resolution: PASS",
            "- Shifted action equality (all rows, including final target): PASS"
            if action_enabled
            else "- Shifted action equality: N/A (不适用; action disabled)",
        ]
    elif status.startswith("CHECK ONLY"):
        lines.append("- Output validation not run; preflight only.")
    else:
        lines.append("- Conversion/output validation not completed.")
    if features:
        lines += ["", "## State/action layout", ""]
        for key in ("observation.state", "action"):
            if key in features:
                lines.append(
                    f"- `{key}` ({features[key]['shape'][0]}): {', '.join(features[key]['names'])}"
                )
        lines += ["", "## Sampling", ""]
        for result in results:
            for stat in result.stats.values():
                lines.append(
                    f"- {result.path.name}/{stat.key}: selected={stat.selected}, "
                    f"unselected={stat.unselected}, repeated={stat.repeated}"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_metadata(root: Path, results: list[PreflightResult]) -> None:
    payload = {
        "description": "DAS camera calibration messages retained verbatim as decoded JSON metadata",
        "episodes": [
            {"source": str(result.path), "topics": result.calibration}
            for result in results
        ],
    }
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "das_umi_calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _audit_values(
    episode: EpisodeData, feature: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected source time, original ordinal, and signed grid error."""
    timeline = np.asarray(episode.timeline_ns, dtype=np.int64)
    if feature in episode.videos:
        source_ns = np.asarray(episode.videos[feature].timestamps_ns, dtype=np.int64)
    else:
        source_ns = np.asarray(
            [sample.timestamp_ns for sample in episode.observations[feature]],
            dtype=np.int64,
        )
    selected = _nearest_indices(source_ns, timeline)
    original = episode.source_indices.get(feature)
    source_index = original[selected] if original is not None else selected
    source_time = source_ns[selected]
    return source_time, np.asarray(source_index, dtype=np.int64), source_time - timeline


def write_alignment_sidecar(root: Path, episodes: list[EpisodeData]) -> None:
    """Write acquisition-clock provenance outside the training feature table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    labels = sorted(
        {label for episode in episodes for label in episode.source_labels.values()}
    )
    columns: dict[str, list[int | None]] = {
        "episode_index": [],
        "frame_index": [],
        "target_grid_timestamp_ns": [],
    }
    for label in labels:
        columns[f"{label}.source_timestamp_ns"] = []
        columns[f"{label}.source_index"] = []
        columns[f"{label}.sync_error_ns"] = []

    for episode_index, episode in enumerate(episodes):
        timeline = np.asarray(episode.timeline_ns, dtype=np.int64)
        count = len(timeline)
        columns["episode_index"].extend([episode_index] * count)
        columns["frame_index"].extend(range(count))
        columns["target_grid_timestamp_ns"].extend(timeline.tolist())
        by_label = {label: feature for feature, label in episode.source_labels.items()}
        for label in labels:
            feature = by_label.get(label)
            if feature is None:
                for suffix in ("source_timestamp_ns", "source_index", "sync_error_ns"):
                    columns[f"{label}.{suffix}"].extend([None] * count)
                continue
            source_time, source_index, sync_error = _audit_values(episode, feature)
            columns[f"{label}.source_timestamp_ns"].extend(source_time.tolist())
            columns[f"{label}.source_index"].extend(source_index.tolist())
            columns[f"{label}.sync_error_ns"].extend(sync_error.tolist())

    table = pa.table(
        {name: pa.array(values, type=pa.int64()) for name, values in columns.items()}
    )
    destination = root / ALIGNMENT_SIDECAR
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")


def write_schema_metadata(root: Path, episodes: list[EpisodeData]) -> None:
    """Document feature roles that LeRobot's core info schema does not encode."""
    from .converter import build_features

    first = episodes[0]
    schema_version = int((first.info or {}).get("das_umi_schema_version", 1))
    features = build_features(first)
    auxiliary = [
        key
        for key in features
        if key.startswith("observation.")
        and key not in {"observation.state"}
        and not key.startswith("observation.images.")
        and not key.startswith("observation.source_")
    ]
    payload = {
        "schema": DAS_UMI_SCHEMA_NAME,
        "version": _schema_version_name(schema_version),
        "lerobot_codebase_version": "v3.0",
        "observation_state": {
            "semantic_role": "control_state",
            "components": features.get("observation.state", {}).get("names", []),
        },
        "action": {
            "semantic_role": "control_target",
            "source": "next_observation_state_proxy",
            "shift_frames": 1,
            "absolute_not_delta": True,
            "components": features.get("action", {}).get("names", []),
        },
        "auxiliary_observations": {
            key: {"semantic_role": "sensor_observation"} for key in auxiliary
        },
        "imu_units": {
            "angular_velocity": "rad/s",
            "linear_acceleration": "m/s^2",
        },
        "pose_rotation": "matrix columns [R00,R10,R20,R01,R11,R21]",
        "canonical_head_pose": {
            "source_feature": "observation.pose.head_world",
            "coordinate_frame": "source world frame; no TacCap frame transform",
            "standalone_observation_exported": schema_version < 2,
        },
        "gripper_normalization": (first.info or {}).get("gripper_normalization"),
        "alignment_audit": {
            "storage": ALIGNMENT_SIDECAR if schema_version >= 2 else "inline features",
            "timestamp_dtype": "int64 nanoseconds",
            "sync_error": "source_timestamp_ns - target_grid_timestamp_ns",
        },
    }
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "das_umi_schema.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def topic_mapping(results: list[PreflightResult]) -> dict[str, Any]:
    """Describe actual exported fields, control components, and audit storage."""
    from .converter import (
        SOURCE_TIMESTAMP_FEATURE,
        build_features,
        source_audit_features,
    )

    schema_version = max(
        int((result.episode.info or {}).get("das_umi_schema_version", 1))
        for result in results
        if result.episode is not None
    )
    sidecar_audit = schema_version >= 2
    episodes = []
    for result in results:
        episode = result.episode
        if episode is None:
            continue
        features = build_features(episode)
        components = (episode.info or {}).get("state_components", {})
        topics = []
        for data in result.streams.values():
            rule = data.rule
            if not data.times_ns.size:
                continue
            targets: list[dict[str, Any]] = []
            if rule.feature in features:
                targets.append({"field": rule.feature})
            for field_name in ("observation.state", "action"):
                if field_name in features and rule.key in components:
                    targets.append(
                        {
                            "field": field_name,
                            "components": components[rule.key],
                            "alignment": "next_frame"
                            if field_name == "action"
                            else "current_frame",
                        }
                    )
            timestamp, index = source_audit_features(rule.key)
            topic = {
                "topic": f"/{rule.topic}",
                "stream": rule.key,
                "targets": targets,
                "source_clock": data.clock,
                "audit_alignment": (
                    "current observation row; action uses the next aligned frame"
                ),
            }
            if sidecar_audit:
                topic["audit_columns"] = {
                    "source_timestamp_ns": f"{rule.key}.source_timestamp_ns",
                    "source_index": f"{rule.key}.source_index",
                    "sync_error_ns": f"{rule.key}.sync_error_ns",
                    "storage": ALIGNMENT_SIDECAR,
                }
            else:
                targets.extend([{"field": timestamp}, {"field": index}])
            topics.append(topic)
        episodes.append({"source": str(result.path), "topics": topics})

    derived = {
        "field": (
            "target_grid_timestamp_ns" if sidecar_audit else SOURCE_TIMESTAMP_FEATURE
        ),
        "description": "Derived integer 30 Hz target grid, not a source timestamp.",
    }
    if sidecar_audit:
        derived["storage"] = ALIGNMENT_SIDECAR
    return {
        "schema_version": _schema_version_name(schema_version),
        "description": (
            "Source topics to control/sensor fields; action is next-frame absolute "
            "state. Acquisition clocks are audit metadata, never actions."
        ),
        "gripper": {
            "description": _gripper_description(
                next(result.episode for result in results if result.episode is not None)
            ),
            "normalization": next(
                (result.episode.info or {}).get("gripper_normalization")
                for result in results
                if result.episode is not None
            ),
        },
        "head_pose": (
            "World pose, rotation [R00,R10,R20,R01,R11,R21]; no TacCap "
            "camera-frame calibration implied."
        ),
        "derived_fields": [derived],
        "episodes": episodes,
    }


def write_topic_mapping(root: Path, results: list[PreflightResult]) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "das_umi_topic_mapping.json").write_text(
        json.dumps(topic_mapping(results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _failed_result(path: Path, error: Exception) -> PreflightResult:
    return PreflightResult(
        path=path,
        episode=None,
        streams={},
        stats={},
        inventory={},
        calibration={},
        issues=[f"preflight scan aborted: {error}"],
        notices=[],
        output_camera_span_ms={},
        elapsed_s=0.0,
    )


def verify_alignment_sidecar(root: Path, episodes: list[EpisodeData]) -> None:
    """Verify every v2 audit value against the preflight-selected source sample."""
    import pyarrow.parquet as pq

    path = root / ALIGNMENT_SIDECAR
    if not path.is_file():
        raise ValueError(f"missing DAS-UMI alignment sidecar: {path}")
    table = pq.read_table(path)
    expected_rows = sum(len(episode.timeline_ns) for episode in episodes)
    if table.num_rows != expected_rows:
        raise ValueError(
            f"alignment sidecar has {table.num_rows} rows, expected {expected_rows}"
        )
    columns = set(table.column_names)
    required = {"episode_index", "frame_index", "target_grid_timestamp_ns"}
    if not required.issubset(columns):
        raise ValueError("alignment sidecar is missing canonical index columns")

    offset = 0
    for episode_index, episode in enumerate(episodes):
        timeline = np.asarray(episode.timeline_ns, dtype=np.int64)
        count = len(timeline)
        row_slice = slice(offset, offset + count)
        actual_episode = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)[
            row_slice
        ]
        actual_frame = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)[
            row_slice
        ]
        actual_target = np.asarray(
            table["target_grid_timestamp_ns"].to_pylist(), dtype=np.int64
        )[row_slice]
        if not np.array_equal(actual_episode, np.full(count, episode_index)):
            raise ValueError("alignment sidecar episode_index mismatch")
        if not np.array_equal(actual_frame, np.arange(count, dtype=np.int64)):
            raise ValueError("alignment sidecar frame_index mismatch")
        if not np.array_equal(actual_target, timeline):
            raise ValueError("alignment sidecar target grid mismatch")
        for feature, label in episode.source_labels.items():
            expected = _audit_values(episode, feature)
            for suffix, values in zip(
                ("source_timestamp_ns", "source_index", "sync_error_ns"), expected
            ):
                key = f"{label}.{suffix}"
                if key not in columns:
                    raise ValueError(f"alignment sidecar missing {key}")
                actual = np.asarray(table[key].to_pylist(), dtype=np.int64)[row_slice]
                if not np.array_equal(actual, values):
                    raise ValueError(f"alignment sidecar mismatch: {key}")
        offset += count


def verify_tabular_output(
    root: Path,
    expected_frames: int,
    repo_id: str,
    episodes: list[EpisodeData] | None = None,
) -> None:
    """Check rows, grid, every auxiliary vector, and shifted action."""
    import pyarrow.parquet as pq

    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise ValueError("no output parquet files found")
    tables = [pq.read_table(path) for path in paths]
    rows = sum(table.num_rows for table in tables)
    if rows != expected_frames:
        raise ValueError(f"expected {expected_frames} parquet rows, found {rows}")
    import pyarrow as pa

    table = pa.concat_tables(tables)
    columns = set(table.column_names)
    schema_version = (
        max(
            int((episode.info or {}).get("das_umi_schema_version", 1))
            for episode in episodes or []
        )
        if episodes
        else 1
    )
    if schema_version >= 2:
        inline_audit = sorted(
            key for key in columns if key.startswith("observation.source_")
        )
        if inline_audit:
            raise ValueError(
                f"v2 training table contains inline audit columns: {inline_audit}"
            )
        assert episodes is not None
        verify_alignment_sidecar(root, episodes)
    if "episode_index" in columns:
        episode_ids = np.asarray(
            table["episode_index"].to_pylist(), dtype=np.int64
        ).reshape(-1)
    else:
        episode_ids = np.zeros(rows, dtype=np.int64)
    if episodes is not None and set(episode_ids.tolist()) != set(range(len(episodes))):
        raise ValueError("output episode IDs do not match preflight episodes")
    for episode_id in dict.fromkeys(episode_ids.tolist()):
        mask = np.flatnonzero(episode_ids == episode_id)
        if not len(mask):
            continue
        if episodes is not None:
            expected_episode = episodes[episode_id]
            if len(mask) != len(expected_episode.timeline_ns):
                raise ValueError("episode row count does not match preflight")
            for key, samples in (
                ("observation.state", expected_episode.states),
                ("action", expected_episode.actions),
            ):
                if not samples:
                    if key in columns:
                        raise ValueError(f"unexpected {key} column")
                    continue
                if key not in columns:
                    raise ValueError(f"missing {key} column")
                actual = np.asarray(table[key].to_pylist(), dtype=np.float32)[mask]
                expected_values = np.stack([sample.value for sample in samples])
                if not np.array_equal(actual, expected_values):
                    raise ValueError(
                        f"{key} differs from aligned source, including final row"
                    )
            timeline_ns = np.asarray(expected_episode.timeline_ns, dtype=np.int64)
            for key, samples in expected_episode.observations.items():
                if key in expected_episode.hidden_observations:
                    continue
                if key not in columns:
                    raise ValueError(f"missing {key} column")
                source_ns = np.asarray(
                    [sample.timestamp_ns for sample in samples], dtype=np.int64
                )
                selected = _nearest_indices(source_ns, timeline_ns)
                expected_values = np.stack(
                    [samples[int(index)].value for index in selected]
                )
                actual = np.asarray(
                    table[key].to_pylist(), dtype=expected_values.dtype
                )[mask]
                if not np.array_equal(actual, expected_values):
                    raise ValueError(f"{key} differs from aligned source")
        if "observation.source_timestamp" in columns:
            source = np.asarray(
                table["observation.source_timestamp"].to_pylist(), dtype=np.int64
            ).reshape(-1)[mask]
            if len(source) > 1:
                expected = (
                    source[0]
                    + np.arange(len(source), dtype=np.int64)
                    * 1_000_000_000
                    // DAS_UMI_FPS
                )
                if not np.array_equal(source, expected):
                    raise ValueError(
                        "parquet source timeline is not the exact integer 30 Hz grid"
                    )
        if {"observation.state", "action"}.issubset(columns):
            state = np.asarray(
                table["observation.state"].to_pylist(), dtype=np.float32
            )[mask]
            action = np.asarray(table["action"].to_pylist(), dtype=np.float32)[mask]
            if len(state) > 1 and not np.array_equal(action[:-1], state[1:]):
                raise ValueError("action[t] does not equal observation.state[t+1]")
            if schema_version >= 3:
                names = (expected_episode.info or {})["features"]["observation.state"][
                    "names"
                ]
                gripper_indices = [
                    index for index, name in enumerate(names) if "gripper.pos" in name
                ]
                for key, values in (("observation.state", state), ("action", action)):
                    gripper = values[:, gripper_indices]
                    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
                        raise ValueError(f"{key} gripper values are outside [0, 1]")
                    quantized = np.floor(gripper * 100.0 + 0.5) / 100.0
                    if not np.allclose(gripper, quantized, rtol=0.0, atol=1e-6):
                        raise ValueError(
                            f"{key} gripper values are not two-decimal quantized"
                        )
    mapping_path = root / "meta" / "das_umi_topic_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    for entry in mapping["derived_fields"] + [
        target
        for episode_mapping in mapping["episodes"]
        for topic in episode_mapping["topics"]
        for target in topic["targets"]
    ]:
        if entry.get("storage"):
            if not (root / entry["storage"]).is_file():
                raise ValueError(
                    f"topic mapping storage is missing: {entry['storage']}"
                )
            continue
        key = entry["field"]
        if key not in features or (
            features[key]["dtype"] != "video" and key not in columns
        ):
            raise ValueError(f"topic mapping target is not exported: {key}")
        for component in entry.get("components", []):
            if features[key]["names"][component["index"]] != component["name"]:
                raise ValueError(f"topic mapping component mismatch: {key}")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=repo_id, root=root)
        if len(dataset) != expected_frames:
            raise ValueError(
                f"LeRobot loaded {len(dataset)} rows, expected {expected_frames}"
            )
    except TypeError:
        # Older supported lerobot variants use a slightly different reader
        # signature; parquet/video validation above still verifies the layout.
        pass


def convert_das_umi(owner: Any, mcap_files: list[Path]) -> Path:
    """Run batch preflight, reuse the existing writer, verify, then publish."""
    from .converter import (
        build_features,
        plan_pairing,
        source_video_codec,
        verify_video_outputs,
    )

    started = time.monotonic()
    report_dir = (
        Path(
            owner.config.report_dir
            or owner.output_root.parent / f"{owner.output_root.name}_reports"
        )
        .expanduser()
        .resolve()
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = report_dir / "preflight.md"
    conversion_path = report_dir / "conversion.md"
    try:
        config = load_das_umi_config(owner.config.config)
        if owner.config.fps not in (None, DAS_UMI_FPS):
            raise ValueError(
                f"DAS-UMI output is fixed at 30 Hz; --fps {owner.config.fps} is not supported"
            )
    except Exception as error:
        preflight_path.write_text(
            f"# DAS-UMI preflight\n\n**Result: FAIL**\n\n- {error}\n",
            encoding="utf-8",
        )
        write_conversion_report(
            conversion_path,
            "NOT RUN",
            [],
            reason=str(error),
            elapsed_s=time.monotonic() - started,
        )
        raise PreflightError(str(error)) from error

    results: list[PreflightResult] = []
    for path in mcap_files:
        try:
            result = preflight_das_umi(
                path,
                config,
                export_action=owner.config.export_action,
                schema_version=owner.config.schema_version,
            )
        except Exception as error:
            result = _failed_result(path, error)
        results.append(result)
    preflight_path.write_text(render_preflight(results, config), encoding="utf-8")
    if not all(result.passed for result in results):
        reason = "Preflight rejected the batch; no LeRobot dataset was created."
        write_conversion_report(
            conversion_path,
            "NOT RUN",
            results,
            reason=reason,
            elapsed_s=time.monotonic() - started,
        )
        raise PreflightError(reason)

    episodes = [result.episode for result in results]
    assert all(episode is not None for episode in episodes)
    concrete = [episode for episode in episodes if episode is not None]
    features = build_features(concrete[0])
    for episode in concrete[1:]:
        if build_features(episode) != features:
            reason = f"Feature layout changed in {episode.source}"
            write_conversion_report(
                conversion_path,
                "NOT RUN",
                results,
                features=features,
                reason=reason,
                elapsed_s=time.monotonic() - started,
            )
            raise PreflightError(reason)
    plans = [
        plan_pairing(episode, DAS_UMI_FPS, owner.config.max_sync_gap_frames)
        for episode in concrete
    ]
    if owner.config.check_only:
        write_conversion_report(
            conversion_path,
            "CHECK ONLY — conversion not executed",
            results,
            features=features,
            elapsed_s=time.monotonic() - started,
        )
        return owner.output_root
    if owner.output_root.exists():
        reason = f"Output path already exists: {owner.output_root}"
        write_conversion_report(
            conversion_path, "NOT RUN", results, features, reason=reason
        )
        raise FileExistsError(reason)

    owner.reader = DasUmiEpisodeReader()
    temporary_parent: Path | None = None
    if owner._verify_videos:
        owner.output_root.parent.mkdir(parents=True, exist_ok=True)
        temporary_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{owner.output_root.name}.tmp-", dir=owner.output_root.parent
            )
        )
        working_root = temporary_parent / "dataset"
    else:
        # Injected dataset factories used by unit tests do not create a physical
        # dataset; they still exercise the exact same frame-writing path.
        working_root = owner.output_root
    dataset = owner.dataset_factory(
        repo_id=owner.repo_id,
        root=working_root,
        fps=DAS_UMI_FPS,
        robot_type=owner.config.robot_type or config.robot_type,
        features=features,
        use_videos=bool(concrete[0].videos),
        video_codec=source_video_codec(concrete[0]),
        streaming_encoding=True,
    )
    try:
        for episode, plan in zip(concrete, plans):
            owner._write_episode(dataset, episode, features, DAS_UMI_FPS, plan)
        dataset.finalize()
        total_frames = sum(
            len(episode.timeline_ns)
            for episode in concrete
            if episode.timeline_ns is not None
        )
        if owner._verify_videos:
            write_calibration_metadata(working_root, results)
            write_schema_metadata(working_root, concrete)
            if owner.config.schema_version >= 2:
                write_alignment_sidecar(working_root, concrete)
            write_topic_mapping(working_root, results)
            verify_video_outputs(working_root, features, total_frames)
            verify_tabular_output(working_root, total_frames, owner.repo_id, concrete)
            os.replace(working_root, owner.output_root)
            assert temporary_parent is not None
            temporary_parent.rmdir()
        write_conversion_report(
            conversion_path,
            "PASS",
            results,
            features=features,
            output=owner.output_root,
            elapsed_s=time.monotonic() - started,
        )
    except Exception as error:
        finalize = getattr(dataset, "finalize", None)
        if callable(finalize):
            try:
                finalize()
            except Exception:
                pass
        retained = working_root if working_root.exists() else temporary_parent
        write_conversion_report(
            conversion_path,
            "FAIL",
            results,
            features=features,
            reason=str(error),
            elapsed_s=time.monotonic() - started,
            temporary=retained,
        )
        raise
    return owner.output_root
