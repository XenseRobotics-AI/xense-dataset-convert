"""Generic LeRobot to Zarr conversion."""

from __future__ import annotations

import json
import math
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .logger import get_logger
from xense_dataset_convert._parallelism import available_cpu_count

logger = get_logger("lerobot2zarr")


class FrameDataset(Protocol):
    """Minimal interface required by :class:`LeRobotZarrConverter`."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FeatureSpec:
    """Storage information inferred for one LeRobot feature."""

    source_name: str
    zarr_path: str
    shape: tuple[int, ...]
    dtype: np.dtype
    kind: str
    source_dtype: str | None
    transform: str | None = None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _image_to_uint8_hwc(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim != 3:
        raise ValueError(f"Image feature must have 3 dimensions, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).round().astype(np.uint8))


def _feature_path(feature_name: str) -> str:
    parts = feature_name.split(".")
    if any(not part or "/" in part for part in parts):
        raise ValueError(f"Feature name cannot be mapped safely to Zarr: {feature_name!r}")
    return "/".join(parts)


def _string_dtype(values: Iterable[Any]) -> np.dtype:
    max_length = max(1, *(len(str(value)) for value in values))
    return np.dtype(f"<U{max_length}")


class LeRobotZarrConverter:
    """Write a dataset's frame-aligned LeRobot features to a generic Zarr store.

    Feature names are preserved in ``meta/schema.json`` and mapped to nested Zarr
    paths by replacing dots with slashes. Video/image frames are normalized to
    contiguous HWC ``uint8`` arrays; other numeric features retain their shape and
    dtype.
    """

    def __init__(
        self,
        dataset: FrameDataset,
        output_path: Path,
        feature_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        source: Mapping[str, Any] | None = None,
        chunk_bytes: int = 8 * 1024 * 1024,
        workers: int = 0,
    ):
        self.dataset = dataset
        self.output_path = Path(output_path)
        self.feature_metadata = dict(feature_metadata or {})
        self.source = dict(source or {})
        self.chunk_bytes = chunk_bytes
        if workers < 0:
            raise ValueError("workers must be 0 (auto) or positive")
        self.workers = workers

    def convert(self, limit: int | None = None) -> Path:
        import zarr
        from numcodecs import Blosc

        total_frames = len(self.dataset)
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be greater than zero")
            total_frames = min(total_frames, limit)
        if total_frames == 0:
            raise ValueError("Dataset contains no frames")

        first_row = self.dataset[0]
        feature_names = self._ordered_feature_names(first_row)
        specs = self._infer_specs(feature_names, first_row, total_frames)
        self._validate_unique_paths(specs)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(self.output_path), mode="w")
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")
        arrays = {}
        for spec in specs:
            arrays[spec.source_name] = data_group.create_dataset(
                spec.zarr_path,
                shape=(total_frames, *spec.shape),
                chunks=self._chunks(spec, total_frames),
                dtype=spec.dtype,
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
            )

        episode_indices = np.empty(total_frames, dtype=np.int64)
        worker_count = min(self.workers or available_cpu_count(), len(specs))
        executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
        write_locks = (
            {spec.source_name: Lock() for spec in specs} if executor is not None else {}
        )
        pending: set[Future[None]] = set()
        pending_limit = max(1, worker_count * 2)
        logger.info(
            f"Writing {len(specs)} features with {worker_count} worker(s)"
        )
        try:
            for index in range(total_frames):
                row = first_row if index == 0 else self.dataset[index]
                for spec in specs:
                    if spec.source_name not in row:
                        raise KeyError(
                            f"Feature {spec.source_name!r} is missing from frame {index}"
                        )
                    if executor is None:
                        self._write_feature_value(
                            arrays[spec.source_name], spec, row[spec.source_name], index
                        )
                    else:
                        while len(pending) >= pending_limit:
                            self._wait_for_write(pending)
                        pending.add(
                            executor.submit(
                                self._write_feature_value,
                                arrays[spec.source_name],
                                spec,
                                row[spec.source_name],
                                index,
                                write_locks[spec.source_name],
                            )
                        )
                episode_indices[index] = self._episode_index(row, index)
            while pending:
                self._wait_for_write(pending)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        episode_ends, episode_ids = self._episode_boundaries(episode_indices)
        meta_group.create_dataset(
            "episode_ends",
            data=episode_ends,
            chunks=(max(1, len(episode_ends)),),
        )
        meta_group.create_dataset(
            "episode_ids",
            data=episode_ids,
            chunks=(max(1, len(episode_ids)),),
        )

        schema = {
            "format": "lerobot-generic-zarr",
            "format_version": 1,
            "frame_count": total_frames,
            "episode_count": len(episode_ends),
            "feature_path_encoding": "dots-to-groups",
            "features": {
                spec.source_name: {
                    "path": f"data/{spec.zarr_path}",
                    "shape": list(spec.shape),
                    "dtype": str(spec.dtype),
                    "kind": spec.kind,
                    "source_dtype": spec.source_dtype,
                    "transform": spec.transform,
                }
                for spec in specs
            },
            "source": self.source,
        }
        root.attrs.update(
            {
                "format": schema["format"],
                "format_version": schema["format_version"],
                "frame_count": total_frames,
                "episode_count": len(episode_ends),
            }
        )
        (self.output_path / "meta" / "schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            f"Converted {total_frames} frames and {len(specs)} features to {self.output_path}"
        )
        return self.output_path

    def _ordered_feature_names(self, first_row: Mapping[str, Any]) -> list[str]:
        names = list(self.feature_metadata)
        names.extend(name for name in first_row if name not in self.feature_metadata)
        return names

    def _infer_specs(
        self,
        feature_names: Sequence[str],
        first_row: Mapping[str, Any],
        total_frames: int,
    ) -> list[FeatureSpec]:
        specs = []
        for feature_name in feature_names:
            if feature_name not in first_row:
                logger.warn(f"Skipping metadata-only feature absent from frames: {feature_name}")
                continue
            metadata = self.feature_metadata.get(feature_name, {})
            source_dtype = metadata.get("dtype")
            value = first_row[feature_name]
            if source_dtype in {"video", "image"}:
                array = _image_to_uint8_hwc(value)
                specs.append(
                    FeatureSpec(
                        feature_name,
                        _feature_path(feature_name),
                        array.shape,
                        array.dtype,
                        source_dtype,
                        source_dtype,
                        "chw-or-hwc-to-hwc-uint8",
                    )
                )
                continue

            array = _to_numpy(value)
            if array.dtype.kind in "biufc":
                specs.append(
                    FeatureSpec(
                        feature_name,
                        _feature_path(feature_name),
                        array.shape,
                        array.dtype,
                        "numeric",
                        source_dtype,
                    )
                )
            elif array.dtype.kind in "US" or isinstance(value, str):
                samples = [first_row[feature_name]]
                samples.extend(self.dataset[i][feature_name] for i in range(1, total_frames))
                dtype = _string_dtype(samples)
                specs.append(
                    FeatureSpec(
                        feature_name,
                        _feature_path(feature_name),
                        array.shape,
                        dtype,
                        "string",
                        source_dtype,
                    )
                )
            else:
                raise TypeError(
                    f"Feature {feature_name!r} has unsupported dtype {array.dtype}; "
                    "only numeric, boolean, string, image, and video features are supported"
                )
        return specs

    @staticmethod
    def _validate_unique_paths(specs: Sequence[FeatureSpec]) -> None:
        paths = [spec.zarr_path for spec in specs]
        if len(paths) != len(set(paths)):
            raise ValueError("Multiple feature names map to the same Zarr path")

    def _chunks(self, spec: FeatureSpec, total_frames: int) -> tuple[int, ...]:
        bytes_per_frame = max(1, math.prod(spec.shape or (1,)) * spec.dtype.itemsize)
        frame_chunk = max(1, min(total_frames, self.chunk_bytes // bytes_per_frame))
        if spec.kind in {"video", "image"}:
            frame_chunk = 1
        return (frame_chunk, *spec.shape)

    @staticmethod
    def _prepare_value(spec: FeatureSpec, value: Any) -> np.ndarray:
        if spec.kind in {"video", "image"}:
            return _image_to_uint8_hwc(value)
        return _to_numpy(value).astype(spec.dtype, copy=False)

    def _write_feature_value(
        self,
        array: Any,
        spec: FeatureSpec,
        value: Any,
        index: int,
        write_lock: Any | None = None,
    ) -> None:
        prepared = self._prepare_value(spec, value)
        if prepared.shape != spec.shape:
            raise ValueError(
                f"Feature {spec.source_name!r} changed shape at frame {index}: "
                f"expected {spec.shape}, got {prepared.shape}"
            )
        if write_lock is None:
            array[index] = prepared
        else:
            with write_lock:
                array[index] = prepared

    @staticmethod
    def _wait_for_write(pending: set[Future[None]]) -> None:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        pending.difference_update(completed)
        for future in completed:
            future.result()

    @staticmethod
    def _episode_index(row: Mapping[str, Any], frame_index: int) -> int:
        if "episode_index" not in row:
            return 0
        value = _to_numpy(row["episode_index"]).reshape(-1)
        if value.size != 1:
            raise ValueError(
                f"episode_index must be scalar at frame {frame_index}, got {value.shape}"
            )
        return int(value[0])

    @staticmethod
    def _episode_boundaries(episode_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        changes = np.flatnonzero(np.diff(episode_indices) != 0) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(episode_indices)])).astype(np.int64)
        ids = episode_indices[starts].astype(np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Frames for the same episode are not contiguous")
        return ends, ids


def load_info_json(dataset_root: Path) -> dict[str, Any]:
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def open_lerobot_dataset(
    dataset_root: Path,
    repo_id: str | None = None,
    episodes: list[int] | None = None,
) -> FrameDataset:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(dataset_root).expanduser().resolve()
    return LeRobotDataset(
        repo_id=repo_id or dataset_root.name,
        root=dataset_root,
        episodes=episodes,
    )
