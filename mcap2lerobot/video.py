"""Decode Foxglove CompressedVideo packets."""

from __future__ import annotations

from typing import Any

import numpy as np


class VideoStreamDecoder:
    """Decode the independently decodable packets written by lerobot2mcap."""

    def __init__(self, codec_name: str):
        try:
            import av
        except ImportError as error:
            raise ImportError("PyAV is required to decode MCAP video packets") from error
        self._av: Any = av
        self._codec = av.CodecContext.create(codec_name, "r")

    def decode(self, data: bytes) -> np.ndarray:
        frames = self._codec.decode(self._av.Packet(data))
        if len(frames) != 1:
            raise ValueError(
                "Expected one decoded image per CompressedVideo message, "
                f"got {len(frames)}"
            )
        return np.ascontiguousarray(frames[0].to_ndarray(format="rgb24"))
