"""Orchestrate MCAP episode conversion into a LeRobot v3 dataset."""

from __future__ import annotations

import copy
import inspect
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .config import DEFAULT_ROBOT_TYPE, DEFAULT_TASK, ConverterConfig
from .reader import EpisodeData, McapEpisodeReader, TimedArray, TimedImage

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
            files.extend(path.glob("*.mcap"))
        elif path.is_file() and path.suffix.lower() == ".mcap":
            files.append(path)
        else:
            raise FileNotFoundError(f"MCAP input does not exist: {path}")
    unique_files = list(dict.fromkeys(files))
    return sorted(unique_files, key=_episode_sort_key)


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 2**63 - 1, path.name)


def infer_fps(episode: EpisodeData) -> int:
    timestamps = np.asarray([item.timestamp_ns for item in episode.states], dtype=np.int64)
    if timestamps.size < 2:
        raise ValueError("Cannot infer fps from an episode with fewer than two states")
    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]
    if intervals.size == 0:
        raise ValueError("Cannot infer fps from duplicate state timestamps")
    fps = int(round(1_000_000_000 / float(np.median(intervals))))
    if fps <= 0:
        raise ValueError("Inferred fps is invalid")
    return fps


def nearest_value(
    values: Sequence[TimedArray] | Sequence[TimedImage],
    timestamp_ns: int,
    max_gap_ns: int,
    label: str,
) -> np.ndarray:
    timestamps = np.asarray([item.timestamp_ns for item in values], dtype=np.int64)
    insertion = int(np.searchsorted(timestamps, timestamp_ns))
    candidates = []
    if insertion < len(values):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    if not candidates:
        raise ValueError(f"No samples available for {label}")
    index = min(candidates, key=lambda candidate: abs(int(timestamps[candidate]) - timestamp_ns))
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


def _vector_feature(episode: EpisodeData, key: str, value: np.ndarray, prefix: str) -> dict[str, Any]:
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
                "names": names if names is not None else _generated_names(prefix, value.size),
            }
        logger.warning(
            "Ignoring the embedded %s declaration (dtype=%s, shape=%s): the MCAP carries "
            "vectors of %d values",
            key, dtype, shape, value.size,
        )
    return {
        "dtype": value.dtype.name,
        "shape": value.shape,
        "names": _generated_names(prefix, value.size),
    }


def _video_feature(episode: EpisodeData, key: str, image: np.ndarray) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "dtype": "video",
        "shape": image.shape,
        "names": ["height", "width", "channels"],
    }
    declared = declared_feature(episode, key)
    if declared is not None:
        if declared.get("dtype") in ("video", "image") and tuple(declared.get("shape") or ()) == tuple(
            image.shape
        ):
            entry["names"] = declared.get("names") or entry["names"]
        else:
            logger.warning(
                "Ignoring the embedded %s declaration (dtype=%s, shape=%s): decoded frames are %s",
                key, declared.get("dtype"), declared.get("shape"), image.shape,
            )
    return entry


def build_features(episode: EpisodeData) -> dict[str, dict[str, Any]]:
    state = episode.states[0].value
    action = episode.actions[0].value
    features: dict[str, dict[str, Any]] = {
        "observation.state": _vector_feature(episode, "observation.state", state, "state"),
        "action": _vector_feature(episode, "action", action, "action"),
    }
    for feature_name, images in sorted(episode.videos.items()):
        image = images[0].value
        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            raise ValueError(f"{feature_name} must be an HWC image, got {image.shape}")
        features[feature_name] = _video_feature(episode, feature_name, image)
    return features


def resolve_fps(config: ConverterConfig, episode: EpisodeData) -> int:
    """Explicit override, else the source dataset's fps, else inferred."""
    if config.fps:
        return config.fps
    embedded = (episode.info or {}).get("fps")
    if embedded:
        return int(embedded)
    return infer_fps(episode)


def resolve_robot_type(config: ConverterConfig, episode: EpisodeData) -> str:
    return config.robot_type or str((episode.info or {}).get("robot_type") or DEFAULT_ROBOT_TYPE)


def source_video_codec(episode: EpisodeData) -> str | None:
    """The codec the source dataset's videos were encoded with, per ``info.json``."""
    features = (episode.info or {}).get("features") or {}
    for spec in features.values():
        if isinstance(spec, dict) and spec.get("dtype") == "video":
            codec = (spec.get("info") or {}).get("video.codec")
            if codec:
                return str(codec)
    return None


def frame_task(config: ConverterConfig, episode: EpisodeData, state: TimedArray) -> str:
    """Task text for one frame: override, else the row's own task, else default."""
    if config.task:
        return config.task
    meta = episode.episode_meta or {}
    mapping = meta.get("task_index_to_task") or {}
    if state.task_index is not None and str(state.task_index) in mapping:
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
        logger.warning("Source videos use %r, which lerobot cannot encode; using its default", video_codec)
        return {}
    parameters = inspect.signature(dataset_cls.create).parameters
    if "vcodec" in parameters:
        return {"vcodec": encoder}
    if "rgb_encoder" in parameters:
        from lerobot.configs.video import RGBEncoderConfig

        return {"rgb_encoder": RGBEncoderConfig(vcodec=encoder)}
    logger.warning("This lerobot exposes no encoder setting on LeRobotDataset.create; using its default codec")
    return {}


class McapToLeRobotConverter:
    """Convert one MCAP file per episode into one LeRobot v3 dataset."""

    def __init__(
        self,
        inputs: Sequence[Path],
        output_root: Path,
        repo_id: str,
        config: ConverterConfig | None = None,
        reader: McapEpisodeReader | None = None,
        dataset_factory: Callable[..., DatasetWriter] | None = None,
    ):
        self.inputs = [Path(path) for path in inputs]
        self.output_root = Path(output_root).expanduser().resolve()
        self.repo_id = repo_id
        self.config = config or ConverterConfig()
        self.reader = reader or McapEpisodeReader(self.config)
        self.dataset_factory = dataset_factory or self._default_dataset_factory

    @staticmethod
    def _default_dataset_factory(video_codec: str | None = None, **kwargs: Any) -> DatasetWriter:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise ImportError("lerobot is required to write the dataset") from error
        # lerobot keeps a reference to the `features` dict it is given and writes
        # each video's probed `info` block into it after the first episode is
        # saved. Hand it a copy so `_write_episode`'s layout check keeps comparing
        # against the layout we built, not against lerobot's annotated one.
        kwargs["features"] = copy.deepcopy(kwargs["features"])
        kwargs.update(encoder_kwargs(LeRobotDataset, video_codec))
        return LeRobotDataset.create(**kwargs)

    def convert(self) -> Path:
        mcap_files = discover_mcap_files(self.inputs)
        if not mcap_files:
            raise ValueError("No .mcap files found")
        if self.output_root.exists():
            raise FileExistsError(f"Output path already exists: {self.output_root}")

        first_episode = self.reader.read(mcap_files[0])
        fps = resolve_fps(self.config, first_episode)
        features = build_features(first_episode)
        robot_type = resolve_robot_type(self.config, first_episode)
        video_codec = source_video_codec(first_episode)
        if first_episode.info is None:
            logger.info(
                "%s carries no embedded LeRobot metadata; feature names, task and "
                "robot_type fall back to defaults",
                mcap_files[0].name,
            )
        logger.info(
            "Creating LeRobot v3 dataset at %s (fps=%d, robot_type=%s)", self.output_root, fps, robot_type
        )
        dataset = self.dataset_factory(
            repo_id=self.repo_id,
            root=self.output_root,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=bool(first_episode.videos),
            video_codec=video_codec,
        )

        try:
            self._write_episode(dataset, first_episode, features, fps)
            for mcap_path in mcap_files[1:]:
                episode = self.reader.read(mcap_path)
                self._write_episode(dataset, episode, features, fps)
            dataset.finalize()
        except Exception:
            finalize = getattr(dataset, "finalize", None)
            if callable(finalize):
                try:
                    finalize()
                except Exception:
                    pass
            raise
        return self.output_root

    def _write_episode(
        self,
        dataset: DatasetWriter,
        episode: EpisodeData,
        expected_features: dict[str, dict[str, Any]],
        fps: int,
    ) -> None:
        logger.info("Converting episode from %s", episode.source)
        actual_features = build_features(episode)
        if actual_features != expected_features:
            raise ValueError(f"Feature layout changed in {episode.source}")

        state_dtype = np.dtype(expected_features["observation.state"]["dtype"])
        action_dtype = np.dtype(expected_features["action"]["dtype"])
        max_gap_ns = int(self.config.max_sync_gap_frames * 1_000_000_000 / fps)
        for state in episode.states:
            # No "timestamp" key: lerobot's validate_frame rejects it as an extra
            # feature, and add_frame derives frame_index / fps itself, which is
            # exactly the value we would pass.
            frame: dict[str, Any] = {
                "observation.state": np.array(state.value, dtype=state_dtype),
                "action": np.array(
                    nearest_value(episode.actions, state.timestamp_ns, max_gap_ns, "action"),
                    dtype=action_dtype,
                ),
                "task": frame_task(self.config, episode, state),
            }
            for feature_name, images in episode.videos.items():
                frame[feature_name] = nearest_value(
                    images, state.timestamp_ns, max_gap_ns, feature_name
                ).copy()
            dataset.add_frame(frame)
        dataset.save_episode()
        logger.info("Saved episode with %d frames", len(episode.states))
