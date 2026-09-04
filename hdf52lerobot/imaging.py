"""Per-camera frame transforms.

A recorder does not always write frames the way the dataset wants them. The
Xense tactile SDK hands out BGR images in the sensor's own portrait
orientation; the capture path in ``xense-taccap-lerobot`` turns those into RGB
and lays them down (``cameras/xense/camera_xense.py``: ``np.transpose(data,
(1, 0, ...))`` on read, then ``arr[..., ::-1]`` when formatting), which is why
every TacVerse dataset carries tactile frames as ``(400, 700, 3)``. A rig that
recorded straight from the SDK skipped both steps, so the converter has to
apply them instead -- per camera, since only the tactile streams need it.

Named in the stream rules:

    - pattern: '.*_tactile_.*\\.mkv$'
      kind: video
      key: '{side}_tactile_{pos}'
      transform: [bgr_to_rgb, transpose]
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Reverse the channel order of an ``(..., 3)`` frame."""
    if frame.ndim < 3 or frame.shape[-1] != 3:
        raise ValueError(f"bgr_to_rgb expects a 3-channel frame, got shape {frame.shape}")
    return np.ascontiguousarray(frame[..., ::-1])


def transpose(frame: np.ndarray) -> np.ndarray:
    """Swap height and width, leaving the channels alone (HWC -> WHC)."""
    if frame.ndim < 2:
        raise ValueError(f"transpose expects an image, got shape {frame.shape}")
    return np.ascontiguousarray(
        np.transpose(frame, (1, 0) + tuple(range(2, frame.ndim)))
    )


TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "bgr_to_rgb": bgr_to_rgb,
    "transpose": transpose,
}

TRANSFORM_NAMES: tuple[str, ...] = tuple(TRANSFORMS)

# Transforms that exchange the frame's width and height, so the feature shape
# and the encoder have to be told the size *after* the transform.
_SIZE_SWAPPING = frozenset({"transpose"})


def validate_transforms(names: Iterable[str]) -> tuple[str, ...]:
    """Check every name is known; returns them as a tuple."""
    resolved = tuple(names)
    unknown = [name for name in resolved if name not in TRANSFORMS]
    if unknown:
        raise ValueError(
            f"Unknown frame transform(s) {unknown}, expected one of {TRANSFORM_NAMES}"
        )
    return resolved


def compose(names: Sequence[str]) -> Callable[[np.ndarray], np.ndarray] | None:
    """One callable applying ``names`` in order, or None when there are none."""
    steps = [TRANSFORMS[name] for name in validate_transforms(names)]
    if not steps:
        return None

    def apply(frame: np.ndarray) -> np.ndarray:
        for step in steps:
            frame = step(frame)
        return frame

    return apply


def transformed_size(width: int, height: int, names: Sequence[str]) -> tuple[int, int]:
    """``(width, height)`` after the transforms, for features and the encoder."""
    for name in validate_transforms(names):
        if name in _SIZE_SWAPPING:
            width, height = height, width
    return int(width), int(height)
