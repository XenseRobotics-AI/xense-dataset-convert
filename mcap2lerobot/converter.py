"""Orchestrate MCAP episode conversion into a LeRobot v3 dataset."""

from __future__ import annotations

import copy
import inspect
import logging
import math
import re
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import DEFAULT_ROBOT_TYPE, DEFAULT_TASK, DIALECT_XTAC, ConverterConfig
from .reader import EpisodeData, McapEpisodeReader, TimedArray
from .das_ego import (
    DAS_EGO_ROBOT_TYPE,
    DAS_EGO_TASK,
    SOURCE_TIMESTAMP_FEATURE,
)

logger = logging.getLogger(__name__)

# meta/info.json spells codecs the way ffprobe reports them; lerobot's encoder
# settings take these names (h264 and hevc as-is, av1 as its software encoder).
VIDEO_CODEC_TO_ENCODER = {"h264": "h264", "hevc": "hevc", "av1": "libsvtav1"}


class DatasetWriter(Protocol):
    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def finalize(self) -> None: ...


def discover_mcap_files(inputs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for input_path in inputs:
        path = Path(input_path).expanduser().resolve()
        if path.is_dir():
            # Skip dot-files: a directory copied off macOS carries "._EP*.mcap"
            # AppleDouble side-cars that match the glob but are not recordings.
            files.extend(child for child in path.glob("*.mcap") if not child.name.startswith("."))
        elif path.is_file() and path.suffix.lower() == ".mcap":
            files.append(path)
        else:
            raise FileNotFoundError(f"MCAP input does not exist: {path}")
    unique_files = list(dict.fromkeys(files))
    return sorted(unique_files, key=_episode_sort_key)


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 2**63 - 1, path.name)


def episode_timestamps(episode: EpisodeData) -> np.ndarray:
    if episode.timeline_ns is not None:
        return np.asarray(episode.timeline_ns, dtype=np.int64)
    return np.asarray([item.timestamp_ns for item in episode.states], dtype=np.int64)


def infer_fps(episode: EpisodeData) -> int:
    timestamps = episode_timestamps(episode)
    if timestamps.size < 2:
        raise ValueError("Cannot infer fps from an episode with fewer than two frames")
    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        raise ValueError("Cannot infer fps from duplicate state timestamps")
    fps = int(round(1_000_000_000 / float(np.median(intervals))))
    if fps <= 0:
        raise ValueError("Inferred fps is invalid")
    return fps


def nearest_indices(source_ns: np.ndarray, target_ns: np.ndarray) -> np.ndarray:
    """For each target time, the index of the nearest source time.

    Vectorised twin of :func:`nearest_value` with the same tie rule: when a
    target sits exactly between two sources the later one wins, which is what
    ``min()`` over ``[insertion, insertion - 1]`` picked.
    """
    source = np.asarray(source_ns, dtype=np.int64)
    target = np.asarray(target_ns, dtype=np.int64)
    if source.size == 0:
        raise ValueError("No samples available")
    insertion = np.searchsorted(source, target)  # first source >= target
    right = np.clip(insertion, 0, source.size - 1)
    left = np.clip(insertion - 1, 0, source.size - 1)
    gap_right = np.abs(source[right] - target)
    gap_left = np.abs(source[left] - target)
    take_right = (insertion == 0) | (
        (insertion < source.size) & (gap_right <= gap_left)
    )
    return np.where(take_right, right, left).astype(np.int64)


@dataclass(frozen=True)
class Plan:
    """Which action and which camera packet every state uses: pass two's shopping list."""

    action_index: np.ndarray | None
    """``(states,)`` index into ``episode.actions``."""
    video_index: dict[str, np.ndarray]
    """Feature -> ``(states,)`` packet number within that camera's stream."""
    extra_index: dict[str, np.ndarray] = field(default_factory=dict)
    """Feature -> ``(states,)`` index into that ``episode.extra_vectors`` stream."""
    observation_index: dict[str, np.ndarray] = field(default_factory=dict)
    """Feature -> sample number for additional observation streams such as IMU."""


def _pair_to_states(
    source_ns: np.ndarray, state_ns: np.ndarray, max_gap_ns: int, label: str
) -> np.ndarray:
    source = np.asarray(source_ns, dtype=np.int64)
    if source.size == 0:
        raise ValueError(f"No samples available for {label}")
    index = nearest_indices(source, state_ns)
    gaps = np.abs(source[index] - state_ns)
    worst = int(gaps.argmax())
    if gaps[worst] > max_gap_ns:
        raise ValueError(
            f"Nearest {label} sample is {gaps[worst] / 1e6:.2f} ms away, "
            f"exceeding the {max_gap_ns / 1e6:.2f} ms limit"
        )
    return index


def plan_pairing(episode: EpisodeData, fps: int, max_sync_gap_frames: float) -> Plan:
    """Pair every state with its action and camera packets, before anything is written.

    Raises ``ValueError`` on the first stream that sits further than
    ``max_sync_gap_frames`` from a state, so a badly synchronised episode fails
    before the dataset is created rather than half-way through it.
    """
    state_ns = episode_timestamps(episode)
    max_gap_ns = int(max_sync_gap_frames * 1_000_000_000 / fps)
    action_ns = np.asarray(
        [item.timestamp_ns for item in episode.actions], dtype=np.int64
    )
    return Plan(
        action_index=(
            _pair_to_states(
                action_ns,
                state_ns,
                episode.max_gaps_ns.get("action", max_gap_ns),
                "action",
            )
            if episode.actions
            else None
        ),
        video_index={
            feature_name: _pair_to_states(
                index.timestamps_ns,
                state_ns,
                episode.max_gaps_ns.get(feature_name, max_gap_ns),
                feature_name,
            )
            for feature_name, index in episode.videos.items()
        },
        extra_index={
            feature_name: _pair_to_states(
                np.asarray([item.timestamp_ns for item in samples], dtype=np.int64),
                state_ns,
                max_gap_ns,
                feature_name,
            )
            for feature_name, samples in episode.extra_vectors.items()
        },
        observation_index={
            feature_name: _pair_to_states(
                np.asarray([item.timestamp_ns for item in values], dtype=np.int64),
                state_ns,
                episode.max_gaps_ns.get(feature_name, max_gap_ns),
                feature_name,
            )
            for feature_name, values in episode.observations.items()
        },
    )


def nearest_value(
    values: Sequence[TimedArray],
    timestamp_ns: int,
    max_gap_ns: int,
    label: str,
) -> np.ndarray:
    """Single-query form of the pairing; the converter uses :func:`plan_pairing`."""
    timestamps = np.asarray([item.timestamp_ns for item in values], dtype=np.int64)
    insertion = int(np.searchsorted(timestamps, timestamp_ns))
    candidates = []
    if insertion < len(values):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    if not candidates:
        raise ValueError(f"No samples available for {label}")
    index = min(
        candidates, key=lambda candidate: abs(int(timestamps[candidate]) - timestamp_ns)
    )
    gap = abs(int(timestamps[index]) - timestamp_ns)
    if gap > max_gap_ns:
        raise ValueError(
            f"Nearest {label} sample is {gap / 1e6:.2f} ms away, "
            f"exceeding the {max_gap_ns / 1e6:.2f} ms limit"
        )
    return values[index].value


# -- what the MCAP says about its source dataset ------------------------------


def declared_feature(episode: EpisodeData, key: str) -> dict[str, Any] | None:
    """The source dataset's ``features[key]`` block, when lerobot2mcap attached it."""
    features = (episode.info or {}).get("features") or {}
    spec = features.get(key)
    return spec if isinstance(spec, dict) else None


def _generated_names(prefix: str, size: int) -> list[str]:
    return [f"{prefix}_{index}" for index in range(size)]


def _vector_feature(
    episode: EpisodeData, key: str, value: np.ndarray, prefix: str
) -> dict[str, Any]:
    """Feature entry for a 1-D numeric column, preferring the source declaration.

    The MCAP schema widens or narrows nothing on purpose, but the reader only
    sees a bare numeric vector: the declaration is what restores the original
    dtype and the component names.
    """
    declared = declared_feature(episode, key)
    if declared is not None:
        shape = tuple(int(dim) for dim in (declared.get("shape") or ()))
        dtype = declared.get("dtype")
        names = declared.get("names")
        try:
            numeric = dtype is not None and np.issubdtype(np.dtype(dtype), np.number)
        except TypeError:
            numeric = False
        if numeric and math.prod(shape) == value.size:
            return {
                "dtype": str(dtype),
                "shape": shape,
                "names": names
                if names is not None
                else _generated_names(prefix, value.size),
            }
        logger.warning(
            "Ignoring the embedded %s declaration (dtype=%s, shape=%s): the MCAP carries "
            "vectors of %d values",
            key,
            dtype,
            shape,
            value.size,
        )
    return {
        "dtype": value.dtype.name,
        "shape": value.shape,
        "names": _generated_names(prefix, value.size),
    }


def _video_feature(
    episode: EpisodeData, key: str, shape: tuple[int, ...]
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "dtype": "video",
        "shape": tuple(shape),
        "names": ["height", "width", "channels"],
    }
    declared = declared_feature(episode, key)
    if declared is not None:
        if declared.get("dtype") in ("video", "image") and tuple(
            declared.get("shape") or ()
        ) == tuple(shape):
            entry["names"] = declared.get("names") or entry["names"]
        else:
            logger.warning(
                "Ignoring the embedded %s declaration (dtype=%s, shape=%s): decoded frames are %s",
                key,
                declared.get("dtype"),
                declared.get("shape"),
                tuple(shape),
            )
    return entry


def build_features(episode: EpisodeData) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    if episode.states:
        state = episode.states[0].value
        features["observation.state"] = _vector_feature(
            episode, "observation.state", state, "state"
        )
    if episode.actions:
        action = episode.actions[0].value
        features["action"] = _vector_feature(episode, "action", action, "action")
    for feature_name, samples in sorted(episode.extra_vectors.items()):
        features[feature_name] = _vector_feature(
            episode, feature_name, samples[0].value, feature_name
        )
    for feature_name, values in sorted(episode.observations.items()):
        if feature_name in episode.hidden_observations:
            continue
        features[feature_name] = _vector_feature(
            episode, feature_name, values[0].value, feature_name.rsplit(".", 1)[-1]
        )
    # DAS-UMI embeds source-qualified names for each auxiliary sensor. Preserve
    # those declarations: replacing them with the same six bare IMU names makes
    # independent Ego/left/right streams collide in LeRobot consumers. DAS-Ego
    # has no embedded declaration, so give its sole IMU the same unambiguous
    # feature-relative convention here.
    if episode.profile == "das-ego" and "observation.imu" in features:
        features["observation.imu"]["names"] = [
            f"imu.{component}"
            for component in (
                "angular_velocity.x",
                "angular_velocity.y",
                "angular_velocity.z",
                "linear_acceleration.x",
                "linear_acceleration.y",
                "linear_acceleration.z",
            )
        ]
    if episode.profile in ("das-ego", "das-umi"):
        features[SOURCE_TIMESTAMP_FEATURE] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
    for feature_name, index in sorted(episode.videos.items()):
        shape = tuple(index.shape)
        if len(shape) != 3 or shape[2] not in (1, 3, 4):
            raise ValueError(f"{feature_name} must be an HWC image, got {shape}")
        features[feature_name] = _video_feature(episode, feature_name, shape)
    for label in episode.source_labels.values():
        timestamp_key, index_key = source_audit_features(label)
        features[timestamp_key] = {"dtype": "int64", "shape": (1,), "names": None}
        features[index_key] = {"dtype": "int64", "shape": (1,), "names": None}
    return features


def source_audit_features(label: str) -> tuple[str, str]:
    """Names of the actual source timestamp and raw message-index columns."""
    return (
        f"observation.source_timestamp.{label}",
        f"observation.source_index.{label}",
    )


def resolve_fps(config: ConverterConfig, episode: EpisodeData) -> int:
    """Explicit override, else the source dataset's fps, else inferred."""
    if config.fps:
        return config.fps
    embedded = (episode.info or {}).get("fps")
    if embedded:
        return int(embedded)
    return infer_fps(episode)


def resolve_robot_type(config: ConverterConfig, episode: EpisodeData) -> str:
    if episode.profile == "das-umi":
        return config.robot_type or "das_umi"
    if episode.profile == "das-ego":
        return config.robot_type or DAS_EGO_ROBOT_TYPE
    return config.robot_type or str(
        (episode.info or {}).get("robot_type") or DEFAULT_ROBOT_TYPE
    )


def source_video_codec(episode: EpisodeData) -> str | None:
    """The codec the source dataset's videos were encoded with, per ``info.json``."""
    features = (episode.info or {}).get("features") or {}
    for spec in features.values():
        if isinstance(spec, dict) and spec.get("dtype") == "video":
            codec = (spec.get("info") or {}).get("video.codec")
            if codec:
                return str(codec)
    if episode.profile in ("das-ego", "das-umi"):
        codecs = {video.format for video in episode.videos.values()}
        if len(codecs) == 1:
            return codecs.pop()
    return None


def frame_task(
    config: ConverterConfig, episode: EpisodeData, state: TimedArray | None
) -> str:
    """Task text for one frame: override, else the row's own task, else default."""
    if config.task:
        return config.task
    if episode.profile == "das-umi":
        return config.task or str((episode.info or {}).get("task") or "DAS UMI sensor capture")
    if episode.profile == "das-ego":
        return DAS_EGO_TASK
    meta = episode.episode_meta or {}
    mapping = meta.get("task_index_to_task") or {}
    if (
        state is not None
        and state.task_index is not None
        and str(state.task_index) in mapping
    ):
        return str(mapping[str(state.task_index)])
    tasks = meta.get("tasks") or []
    if len(tasks) == 1:
        return str(tasks[0])
    return DEFAULT_TASK


def encoder_kwargs(dataset_cls: Any, video_codec: str | None) -> dict[str, Any]:
    """Ask lerobot to encode with the source codec, through whichever API it has.

    lerobot 0.5.x takes ``LeRobotDataset.create(vcodec=...)``; 0.6.x moved the
    setting into ``rgb_encoder=RGBEncoderConfig(vcodec=...)``. Without a hint
    lerobot picks its own default (AV1 on both), which is what an MCAP without
    the embedded metadata gets.
    """
    if video_codec is None:
        return {}
    encoder = VIDEO_CODEC_TO_ENCODER.get(video_codec)
    if encoder is None:
        logger.warning(
            "Source videos use %r, which lerobot cannot encode; using its default",
            video_codec,
        )
        return {}
    parameters = inspect.signature(dataset_cls.create).parameters
    if "vcodec" in parameters:
        return {"vcodec": encoder}
    if "rgb_encoder" in parameters:
        from lerobot.configs.video import RGBEncoderConfig

        return {"rgb_encoder": RGBEncoderConfig(vcodec=encoder)}
    logger.warning(
        "This lerobot exposes no encoder setting on LeRobotDataset.create; using its default codec"
    )
    return {}


def verify_video_outputs(
    root: Path, features: dict[str, dict[str, Any]], expected_frames: int
) -> None:
    """Decode every output video and reject missing, resized, or dropped frames."""
    try:
        import av
    except ImportError as error:
        raise ImportError("PyAV is required to verify converted videos") from error

    for feature_name, spec in sorted(features.items()):
        if spec.get("dtype") != "video":
            continue
        files = sorted((root / "videos" / feature_name).glob("**/*.mp4"))
        if not files:
            raise ValueError(f"No output video found for {feature_name}")
        count = 0
        observed_shape: tuple[int, int, int] | None = None
        for path in files:
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    shape = (frame.height, frame.width, 3)
                    observed_shape = observed_shape or shape
                    if shape != observed_shape:
                        raise ValueError(
                            f"{feature_name}: video resolution changes within the stream"
                        )
                    count += 1
        expected_shape = tuple(spec["shape"])
        if observed_shape != expected_shape:
            raise ValueError(
                f"{feature_name}: expected video shape {expected_shape}, got {observed_shape}"
            )
        if count != expected_frames:
            raise ValueError(
                f"{feature_name}: expected {expected_frames} video frames, decoded {count}"
            )
        logger.info(
            "Verified %s: %d frames at %dx%d",
            feature_name,
            count,
            observed_shape[1],
            observed_shape[0],
        )


class McapToLeRobotConverter:
    """Convert one MCAP file per episode into one LeRobot v3 dataset."""

    def __init__(
        self,
        inputs: Sequence[Path],
        output_root: Path,
        repo_id: str,
        config: ConverterConfig | None = None,
        reader: Any | None = None,
        dataset_factory: Callable[..., DatasetWriter] | None = None,
    ):
        self.inputs = [Path(path) for path in inputs]
        self.output_root = Path(output_root).expanduser().resolve()
        self.repo_id = repo_id
        self.config = config or ConverterConfig()
        self.reader = reader
        self._verify_videos = dataset_factory is None
        self.dataset_factory = dataset_factory or self._default_dataset_factory

    @staticmethod
    def _default_reader(config: ConverterConfig) -> Any:
        if config.dialect == DIALECT_XTAC:
            from .xtac import XTacMcapReader

            return XTacMcapReader(config)
        return McapEpisodeReader(config)

    @staticmethod
    def _default_dataset_factory(
        video_codec: str | None = None, **kwargs: Any
    ) -> DatasetWriter:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise ImportError("lerobot is required to write the dataset") from error
        # lerobot keeps a reference to the `features` dict it is given and writes
        # each video's probed `info` block into it after the first episode is
        # saved. Hand it a copy so `_write_episode`'s layout check keeps comparing
        # against the layout we built, not against lerobot's annotated one.
        kwargs["features"] = copy.deepcopy(kwargs["features"])
        parameters = inspect.signature(LeRobotDataset.create).parameters
        if kwargs.get("streaming_encoding") and "streaming_encoding" not in parameters:
            raise RuntimeError(
                "DAS-Ego conversion requires LeRobot streaming video encoding support"
            )
        if "streaming_encoding" not in parameters:
            kwargs.pop("streaming_encoding", None)
        kwargs.update(encoder_kwargs(LeRobotDataset, video_codec))
        return LeRobotDataset.create(**kwargs)

    def convert(self) -> Path:
        mcap_files = discover_mcap_files(self.inputs)
        if not mcap_files:
            raise ValueError("No .mcap files found")
        profile = None
        if self.reader is None:
            from .das_ego import detect_mcap_profile, reader_for_profile

            requested_profile = self.config.profile
            if requested_profile == "auto":
                detected = detect_mcap_profile(mcap_files[0])
                if detected == "ros2":
                    profile = self.config.dialect
                    self.reader = self._default_reader(self.config)
                else:
                    profile, self.reader = reader_for_profile(self.config, mcap_files[0])
            else:
                profile, self.reader = reader_for_profile(self.config, mcap_files[0])
            logger.info("Using %s MCAP source", profile)
        if profile == "das-umi":
            from .das_umi import convert_das_umi

            return convert_das_umi(self, mcap_files)
        if self.config.check_only:
            raise ValueError("--check-only is currently supported by the das-umi profile")
        if self.output_root.exists():
            raise FileExistsError(f"Output path already exists: {self.output_root}")
        first_episode = self.reader.read(mcap_files[0])
        fps = resolve_fps(self.config, first_episode)
        features = build_features(first_episode)
        # Pair before creating anything: a badly synchronised first episode
        # then fails without leaving a half-written dataset behind.
        first_plan = plan_pairing(first_episode, fps, self.config.max_sync_gap_frames)
        robot_type = resolve_robot_type(self.config, first_episode)
        video_codec = source_video_codec(first_episode)
        if first_episode.info is None and first_episode.profile == "ros2":
            logger.info(
                "%s carries no embedded LeRobot metadata; feature names, task and "
                "robot_type fall back to defaults",
                mcap_files[0].name,
            )
        logger.info(
            "Creating LeRobot v3 dataset at %s (fps=%d, robot_type=%s)",
            self.output_root,
            fps,
            robot_type,
        )
        dataset = self.dataset_factory(
            repo_id=self.repo_id,
            root=self.output_root,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=bool(first_episode.videos),
            video_codec=video_codec,
            streaming_encoding=first_episode.profile in ("das-ego", "das-umi"),
        )

        source_metadata: list[dict[str, Any] | None] = [first_episode.source_metadata]
        try:
            self._write_episode(dataset, first_episode, features, fps, first_plan)
            total_frames = len(episode_timestamps(first_episode))
            for mcap_path in mcap_files[1:]:
                episode = self.reader.read(mcap_path)
                self._write_episode(dataset, episode, features, fps)
                source_metadata.append(episode.source_metadata)
                total_frames += len(episode_timestamps(episode))
            dataset.finalize()
            if self._verify_videos and first_episode.profile == "das-ego":
                verify_video_outputs(self.output_root, features, total_frames)
        except Exception:
            finalize = getattr(dataset, "finalize", None)
            if callable(finalize):
                try:
                    finalize()
                except Exception:
                    pass
            raise
        self._write_dialect_metadata(source_metadata, robot_type)
        return self.output_root

    def _write_dialect_metadata(
        self, source_metadata: list[dict[str, Any] | None], robot_type: str
    ) -> None:
        """Write the XTac provenance files the LeRobot writer does not produce."""
        if self.config.dialect != DIALECT_XTAC:
            return
        from .xtac import write_xtac_metadata

        write_xtac_metadata(self.output_root, source_metadata, robot_type)

    def _write_episode(
        self,
        dataset: DatasetWriter,
        episode: EpisodeData,
        expected_features: dict[str, dict[str, Any]],
        fps: int,
        plan: Plan | None = None,
    ) -> None:
        logger.info("Converting episode from %s", episode.source)
        actual_features = build_features(episode)
        if actual_features != expected_features:
            raise ValueError(f"Feature layout changed in {episode.source}")
        if plan is None:
            plan = plan_pairing(episode, fps, self.config.max_sync_gap_frames)

        extra_dtypes = {
            feature_name: np.dtype(expected_features[feature_name]["dtype"])
            for feature_name in episode.extra_vectors
        }
        timeline_ns = episode_timestamps(episode)
        total = len(timeline_ns)
        frames = self.reader.iter_frames(episode, plan.video_index)
        written = 0
        try:
            for position, images in enumerate(frames):
                state = episode.states[position] if episode.states else None
                # No "timestamp" key: lerobot's validate_frame rejects it as an
                # extra feature, and add_frame derives frame_index / fps itself,
                # which is exactly the value we would pass.
                frame: dict[str, Any] = {
                    "task": frame_task(self.config, episode, state),
                }
                for feature_name, samples in episode.extra_vectors.items():
                    sample = samples[int(plan.extra_index[feature_name][position])]
                    frame[feature_name] = np.array(sample.value, dtype=extra_dtypes[feature_name])
                if state is not None:
                    frame["observation.state"] = np.asarray(
                        state.value,
                        dtype=expected_features["observation.state"]["dtype"],
                    )
                if episode.actions:
                    assert plan.action_index is not None
                    frame["action"] = np.asarray(
                        episode.actions[int(plan.action_index[position])].value,
                        dtype=expected_features["action"]["dtype"],
                    )
                for feature_name, values in episode.observations.items():
                    if feature_name in episode.hidden_observations:
                        continue
                    sample = values[int(plan.observation_index[feature_name][position])]
                    frame[feature_name] = np.asarray(
                        sample.value, dtype=expected_features[feature_name]["dtype"]
                    )
                for source_feature, label in episode.source_labels.items():
                    if source_feature in episode.videos:
                        selected_index = int(plan.video_index[source_feature][position])
                        source_time = int(episode.videos[source_feature].timestamps_ns[selected_index])
                    else:
                        selected_index = int(plan.observation_index[source_feature][position])
                        source_time = int(episode.observations[source_feature][selected_index].timestamp_ns)
                    original = episode.source_indices.get(source_feature)
                    source_index = int(original[selected_index]) if original is not None else selected_index
                    timestamp_key, index_key = source_audit_features(label)
                    frame[timestamp_key] = np.asarray([source_time], dtype=np.int64)
                    frame[index_key] = np.asarray([source_index], dtype=np.int64)
                if SOURCE_TIMESTAMP_FEATURE in expected_features:
                    frame[SOURCE_TIMESTAMP_FEATURE] = np.asarray(
                        [timeline_ns[position]], dtype=np.int64
                    )
                frame.update(images)
                dataset.add_frame(frame)
                written += 1
        finally:
            close = getattr(frames, "close", None)
            if callable(close):
                close()
        if written != total:
            raise ValueError(f"{episode.source}: assembled {written} of {total} frames")
        dataset.save_episode()
        logger.info("Saved episode with %d frames", total)
