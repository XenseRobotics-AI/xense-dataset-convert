"""RLDS/TFDS episode conversion to LeRobot v3."""

from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Protocol

import numpy as np

from .config import ConverterConfig
from .reader import episode_count, iter_episodes

logger = logging.getLogger(__name__)


class DatasetWriter(Protocol):
    def add_frame(self, frame: dict[str, Any]) -> None: ...
    def save_episode(self) -> None: ...
    def finalize(self) -> None: ...


def _text(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a numeric vector, got shape {array.shape}")
    return array.astype(np.float32, copy=False).reshape(-1)


def _step_arrays(episode: dict[str, Any]) -> tuple[dict[str, Any], int]:
    steps = episode.get("steps")
    if not isinstance(steps, dict):
        raise ValueError("RLDS episode has no steps dictionary")
    lengths = []
    for value in steps.values():
        if isinstance(value, dict):
            continue
        try:
            lengths.append(len(value))
        except TypeError:
            pass
    if not lengths or len(set(lengths)) != 1:
        raise ValueError(f"RLDS step fields have inconsistent lengths: {lengths}")
    return steps, lengths[0]


def build_features(episode: dict[str, Any], config: ConverterConfig) -> dict[str, dict[str, Any]]:
    steps, length = _step_arrays(episode)
    observation = steps.get("observation") or {}
    if not isinstance(observation, dict):
        raise ValueError("RLDS observation must be a dictionary")
    state_parts = [_vector(observation[field][0], f"observation.{field}") for field in config.state_fields]
    action = _vector(steps[config.action_field][0], f"steps.{config.action_field}")
    state_dim = sum(part.size for part in state_parts)
    state_names = list(config.state_names) or [f"state_{i}" for i in range(state_dim)]
    if len(state_names) != state_dim:
        raise ValueError(f"state_names has {len(state_names)} names, expected {state_dim}")
    action_names = list(config.action_names)
    if len(action_names) != action.size:
        action_names = [f"action_{i}" for i in range(action.size)]
    features: dict[str, dict[str, Any]] = {
        "action": {"dtype": "float32", "shape": [action.size], "names": action_names},
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
    }
    for field in config.image_fields:
        if field not in observation:
            continue
        image = np.asarray(observation[field][0])
        if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"observation.{field} is not an HWC image: {image.shape}")
        features[f"observation.images.{field}"] = {
            "dtype": "video" if config.use_videos else "image",
            "shape": list(image.shape),
            "names": ["height", "width", "channels"],
        }
    if not features.keys() & {"observation.images." + field for field in config.image_fields}:
        logger.warning("No configured image fields found in the first episode")
    return {key: features[key] for key in sorted(features)}


class RldsToLeRobotConverter:
    """Convert generated RLDS/TFDS episodes into one LeRobot dataset."""

    def __init__(
        self,
        inputs: str | Path,
        output_root: str | Path,
        config: ConverterConfig | None = None,
        split: str = "train",
        overwrite: bool = False,
        episode_reader: Callable[[str | Path, str], Iterable[dict[str, Any]]] | None = None,
        dataset_factory: Callable[..., DatasetWriter] | None = None,
    ):
        self.inputs = Path(inputs).expanduser()
        self.output_root = Path(output_root).expanduser().resolve()
        self.config = config or ConverterConfig()
        self.split = split
        self.overwrite = overwrite
        self.episode_reader = episode_reader or iter_episodes
        self.dataset_factory = dataset_factory or self._default_dataset_factory

    @staticmethod
    def _default_dataset_factory(**kwargs: Any) -> DatasetWriter:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as error:
            raise ImportError("lerobot is required to write the dataset") from error
        kwargs["features"] = copy.deepcopy(kwargs["features"])
        requested_vcodec = kwargs.pop("_requested_vcodec", None)
        if requested_vcodec is not None:
            # LeRobot 0.5 accepts vcodec; newer versions may use their default
            # unless an RGBEncoderConfig is supplied. Keep this compatible with
            # the repository's mcap converter by adding it only when supported.
            import inspect
            if "vcodec" in inspect.signature(LeRobotDataset.create).parameters:
                kwargs["vcodec"] = requested_vcodec
        return LeRobotDataset.create(**kwargs)

    def convert(self) -> Path:
        print(
            f"Starting RLDS -> LeRobot conversion: {self.inputs} -> {self.output_root}",
            flush=True,
        )
        if self.output_root.exists():
            if not self.overwrite:
                raise FileExistsError(f"Output already exists: {self.output_root}")
            shutil.rmtree(self.output_root)

        episodes = iter(self.episode_reader(self.inputs, self.split))
        try:
            first = next(episodes)
        except StopIteration:
            raise ValueError("RLDS split contains no episodes") from None
        features = build_features(first, self.config)
        dataset = self.dataset_factory(
            repo_id=self.output_root.name,
            root=self.output_root,
            fps=self.config.fps,
            robot_type=self.config.robot_type,
            features=features,
            use_videos=any(spec["dtype"] == "video" for spec in features.values()),
            _requested_vcodec=self.config.vcodec,
        )
        written = 0
        try:
            episode_total = episode_count(self.inputs, self.split)
            if self.config.max_episodes is not None:
                episode_total = min(episode_total or self.config.max_episodes, self.config.max_episodes)
            episode_iter = self._limited(first, episodes)
            if self.config.show_progress:
                from tqdm import tqdm
                episode_iter = tqdm(episode_iter, total=episode_total, desc="Episodes", unit="episode")
            for episode_index, episode in enumerate(episode_iter, start=1):
                self._write_episode(dataset, episode, features, episode_index)
                written += 1
            if not written:
                raise ValueError("No episodes were converted")
            dataset.finalize()
        except Exception:
            finalize = getattr(dataset, "finalize", None)
            if callable(finalize):
                try:
                    finalize()
                except Exception:
                    pass
            raise
        logger.info("Converted %d episode(s) to %s", written, self.output_root)
        return self.output_root

    def _limited(self, first: dict[str, Any], rest: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        count = 0
        for episode in _chain(first, rest):
            if self.config.max_episodes is not None and count >= self.config.max_episodes:
                return
            yield episode
            count += 1

    def _write_episode(self, dataset: DatasetWriter, episode: dict[str, Any], expected: dict[str, dict[str, Any]], episode_index: int) -> None:
        features = build_features(episode, self.config)
        if features != expected:
            raise ValueError("RLDS feature layout changed between episodes")
        steps, length = _step_arrays(episode)
        observation = steps["observation"]
        task_values = steps.get("language_instruction")
        default_task = self.config.task
        frame_iter = range(length)
        if self.config.show_progress:
            from tqdm import tqdm
            frame_iter = tqdm(frame_iter, total=length, desc=f"Episode {episode_index}", unit="frame", leave=False)
        for i in frame_iter:
            state = np.concatenate([_vector(observation[field][i], field) for field in self.config.state_fields])
            frame: dict[str, Any] = {
                "observation.state": state,
                "action": _vector(steps[self.config.action_field][i], self.config.action_field),
                "task": _text(task_values[i]) if task_values is not None else default_task,
            }
            for field in self.config.image_fields:
                key = f"observation.images.{field}"
                if key in expected:
                    frame[key] = np.asarray(observation[field][i])
            dataset.add_frame(frame)
        dataset.save_episode()


def _chain(first: dict[str, Any], rest: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield first
    yield from rest
