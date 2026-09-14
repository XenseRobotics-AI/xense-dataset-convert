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

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


def _view_bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim < 3 or frame.shape[-1] != 3:
        raise ValueError(f"bgr_to_rgb expects a 3-channel frame, got shape {frame.shape}")
    return frame[..., ::-1]


def _view_transpose(frame: np.ndarray) -> np.ndarray:
    if frame.ndim < 2:
        raise ValueError(f"transpose expects an image, got shape {frame.shape}")
    return np.transpose(frame, (1, 0) + tuple(range(2, frame.ndim)))


# Each step returns a *view*. Copying is what costs -- a 400x700 frame is
# 840 kB and an episode carries thousands of them across four tactile cameras
# -- so the steps compose as views and `compose` materialises the result once,
# which measured 1.7x faster than a copy per step for a bit-identical result.
_STEPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "bgr_to_rgb": _view_bgr_to_rgb,
    "transpose": _view_transpose,
}


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Reverse the channel order of an ``(..., 3)`` frame."""
    return np.ascontiguousarray(_view_bgr_to_rgb(frame))


def transpose(frame: np.ndarray) -> np.ndarray:
    """Swap height and width, leaving the channels alone (HWC -> WHC)."""
    return np.ascontiguousarray(_view_transpose(frame))


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
    steps = [_STEPS[name] for name in validate_transforms(names)]
    if not steps:
        return None

    def apply(frame: np.ndarray) -> np.ndarray:
        for step in steps:
            frame = step(frame)
        # One pass over the pixels, whatever the steps were.
        return np.ascontiguousarray(frame)

    return apply


# Channel order of a frame once the transforms have run: the dataset stores RGB.
CHANNEL_ORDER = ("red", "green", "blue")


def channel_bias(reference: Mapping[str, float], measured: Sequence[float]) -> np.ndarray:
    """``reference - measured`` per channel, in :data:`CHANNEL_ORDER`.

    What the Xense SDK computes at sensor init (2.1.x ``init_rgb_bias``): the
    difference between a reference tone for the sensor model and what this unit
    actually produces, added to every frame so units converge on one tone.
    """
    missing = [name for name in CHANNEL_ORDER if name not in reference]
    if missing:
        raise ValueError(f"rgb reference is missing {missing}; expected {list(CHANNEL_ORDER)}")
    wanted = np.array([float(reference[name]) for name in CHANNEL_ORDER])
    actual = np.asarray(measured, dtype=np.float64).ravel()
    if actual.size != 3:
        raise ValueError(f"expected three channel means, got {actual.size}")
    return wanted - actual


def bias_lookup(bias: np.ndarray) -> np.ndarray:
    """``(3, 256)`` uint8 table applying ``bias`` with saturation.

    A table costs one pass of `np.take` per channel, against a widening add and
    a clip over the whole frame; the SDK builds the same thing for the same
    reason.
    """
    levels = np.arange(256, dtype=np.int32)
    rounded = np.rint(np.asarray(bias, dtype=np.float64)).astype(np.int32)
    return np.clip(levels[None, :] + rounded[:, None], 0, 255).astype(np.uint8)


def apply_bias(frame: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """Apply a :func:`bias_lookup` table to an ``(..., 3)`` uint8 frame."""
    if frame.ndim < 3 or frame.shape[-1] != 3:
        raise ValueError(f"apply_bias expects a 3-channel frame, got shape {frame.shape}")
    out = np.empty_like(frame)
    for channel in range(3):
        np.take(lookup[channel], frame[..., channel], out=out[..., channel])
    return out


def transformed_size(width: int, height: int, names: Sequence[str]) -> tuple[int, int]:
    """``(width, height)`` after the transforms, for features and the encoder."""
    for name in validate_transforms(names):
        if name in _SIZE_SWAPPING:
            width, height = height, width
    return int(width), int(height)
