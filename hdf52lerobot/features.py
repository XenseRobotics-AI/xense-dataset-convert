"""Build the ``features`` block of ``meta/info.json``.

Mirrors the layout of ``/home/xense-5090d/data/TacCap/meta/info.json``: keys
sorted alphabetically, videos carrying a full ``info`` sub-block.
"""

from __future__ import annotations

from typing import Any

from .config import ConverterConfig
from .source import SourceEpisode

SCALAR_FEATURES = ("episode_index", "frame_index", "index", "task_index")
SOURCE_TIMESTAMP = "observation.source_timestamp"

CODEC_NAMES = {"libx264": "h264", "libx265": "hevc", "libsvtav1": "av1"}


def video_info(width: int, height: int, config: ConverterConfig) -> dict[str, Any]:
    return {
        "has_audio": False,
        "video.channels": 3,
        "video.codec": CODEC_NAMES.get(config.vcodec, config.vcodec),
        "video.fps": config.fps,
        "video.height": int(height),
        "video.is_depth_map": False,
        "video.pix_fmt": config.pix_fmt,
        "video.video_backend": "pyav",
        "video.width": int(width),
    }


def build_features(
    episode: SourceEpisode, config: ConverterConfig, state_names: list[str]
) -> dict[str, dict[str, Any]]:
    """Describe every column and video key the dataset will contain."""
    features: dict[str, dict[str, Any]] = {}
    for key in ("action", "observation.state"):
        features[key] = {
            "dtype": "float32",
            "names": list(state_names),
            "shape": [len(state_names)],
        }
    for name in SCALAR_FEATURES:
        features[name] = {"dtype": "int64", "names": None, "shape": [1]}
    features["timestamp"] = {"dtype": "float32", "names": None, "shape": [1]}
    if config.keep_source_timestamp:
        # int64 nanoseconds, not float64 seconds: LeRobot's torch transform turns
        # float scalars into float32, which cannot hold a Unix time to better
        # than a few seconds. int64 survives that path exactly.
        features[SOURCE_TIMESTAMP] = {"dtype": "int64", "names": None, "shape": [1]}

    for stream in episode.videos.values():
        # The shape the dataset carries is the frame *after* the stream's
        # transforms, which can swap width and height.
        width, height = stream.output_size
        features[stream.feature] = {
            "dtype": "video",
            "info": video_info(width, height, config),
            "names": ["height", "width", "channels"],
            "shape": [height, width, 3],
        }

    return {key: features[key] for key in sorted(features)}


def video_features(features: dict[str, dict[str, Any]]) -> list[str]:
    """Feature keys backed by video files, in ``features`` order."""
    return [key for key, spec in features.items() if spec.get("dtype") == "video"]
