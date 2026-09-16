"""Rotation and pose conversions, on the conventions of the Xense lerobot fork.

Ported from ``lerobot.utils.robot_utils`` in xense-taccap-lerobot rather than
imported: that module lives inside a package whose import pulls in torch, and
these converters deliberately run without lerobot installed. What is kept is
its conventions, exactly --

* quaternions are ``[w, x, y, z]`` once inside this module (its
  ``normalize_quaternion`` returns that order whatever it is given);
* the 6D rotation is the first two **columns** of the rotation matrix, laid out
  as ``[m00, m10, m20, m01, m11, m21]`` -- columns, not rows, which is the
  silent-corruption mistake this module exists to prevent;
* a 7D pose is ``[x, y, z, qw, qx, qy, qz]`` and a 9D pose
  ``[x, y, z, r1..r6]``.

Two deliberate differences. The fork's helpers take one pose at a time; these
take ``(N, ...)`` blocks, because an episode is converted as a whole. And these
keep float64, where the fork returns float32 in places: the narrowing belongs at
the writer, once, not part-way through the algebra.
``tests/test_geometry_matches_fork.py`` checks the two agree wherever the fork
can be imported.
"""

from __future__ import annotations

import numpy as np

QUAT_ORDERS = ("wxyz", "xyzw")
"""Component orders a recorder might write a quaternion in."""


def to_wxyz(quaternions: np.ndarray, order: str) -> np.ndarray:
    """Reorder quaternion components into ``[w, x, y, z]``.

    Args:
        quaternions: Array of shape ``(..., 4)``.
        order: Component order of the input, ``"wxyz"`` or ``"xyzw"``.
    """
    if order not in QUAT_ORDERS:
        raise ValueError(f"quaternion order must be one of {QUAT_ORDERS}, got {order!r}")
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.shape[-1] != 4:
        raise ValueError(f"expected trailing dimension 4, got {quaternions.shape}")
    if order == "wxyz":
        return quaternions
    return quaternions[..., [3, 0, 1, 2]]


def quaternions_to_rotation_6d(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Convert ``[w, x, y, z]`` quaternions to the 6D rotation representation.

    Args:
        quaternions_wxyz: Array of shape ``(N, 4)``. Need not be normalized.

    Returns:
        Array of shape ``(N, 6)``: ``[r1, r2, r3]`` is the first column of the
        rotation matrix, ``[r4, r5, r6]`` the second.
    """
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"expected shape (N, 4), got {quaternions.shape}")

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("encountered a zero-norm quaternion")
    w, x, y, z = (quaternions / norms).T

    first_column = np.stack(
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)], axis=1
    )
    second_column = np.stack(
        [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)], axis=1
    )
    return np.concatenate([first_column, second_column], axis=1)


def rotation_6d_to_quaternions(rotations_6d: np.ndarray) -> np.ndarray:
    """Invert :func:`quaternions_to_rotation_6d` (Gram-Schmidt, then to wxyz).

    Only used for verification; the converter itself never needs it.
    """
    rotations = np.asarray(rotations_6d, dtype=np.float64)
    if rotations.ndim != 2 or rotations.shape[1] != 6:
        raise ValueError(f"expected shape (N, 6), got {rotations.shape}")

    a1, a2 = rotations[:, :3], rotations[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    matrices = np.stack([b1, b2, b3], axis=2)  # columns

    trace = np.trace(matrices, axis1=1, axis2=2)
    quaternions = np.empty((len(matrices), 4), dtype=np.float64)
    for index, (matrix, tr) in enumerate(zip(matrices, trace)):
        if tr > 0:
            scale = np.sqrt(tr + 1.0) * 2.0
            quaternions[index] = [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        else:
            diagonal = int(np.argmax(np.diagonal(matrix)))
            i, j, k = diagonal, (diagonal + 1) % 3, (diagonal + 2) % 3
            scale = np.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k]) * 2.0
            components = np.zeros(3)
            components[i] = 0.25 * scale
            components[j] = (matrix[j, i] + matrix[i, j]) / scale
            components[k] = (matrix[k, i] + matrix[i, k]) / scale
            quaternions[index] = [
                (matrix[k, j] - matrix[j, k]) / scale,
                components[0],
                components[1],
                components[2],
            ]

    negative = quaternions[:, 0] < 0
    quaternions[negative] *= -1.0
    return quaternions

def normalize_quaternions(quaternions: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """Unit quaternions in ``[w, x, y, z]``, whatever order came in.

    A zero-norm quaternion becomes the identity rather than a division by zero,
    as in the fork: a recorder that dropped a sample should cost one frame's
    rotation, not the whole episode.
    """
    quaternions = to_wxyz(quaternions, order)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    degenerate = norms < 1e-10
    identity = np.zeros_like(quaternions)
    identity[..., 0] = 1.0
    return np.where(degenerate, identity, quaternions / np.where(degenerate, 1.0, norms))


def quaternions_to_matrices(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """``(N, 4)`` ``[w, x, y, z]`` -> ``(N, 3, 3)`` rotation matrices."""
    quaternions = normalize_quaternions(_as_quaternion_block(quaternions_wxyz))
    w, x, y, z = quaternions.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` -> ``(N, 4)`` ``[w, x, y, z]``.

    Shepperd's method: the component recovered first is the largest one, so the
    division is never by something near zero. The fork branches per matrix; this
    evaluates all four cases and selects, which vectorises.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"expected shape (N, 3, 3), got {matrices.shape}")
    m = matrices
    diagonal = np.stack([m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1)
    trace = diagonal.sum(axis=1)

    candidates = np.stack([trace, diagonal[:, 0], diagonal[:, 1], diagonal[:, 2]], axis=1)
    case = np.argmax(candidates, axis=1)

    # The four branches, each written as [w, x, y, z].
    def branch(index: int) -> np.ndarray:
        if index == 0:
            root = np.sqrt(np.maximum(trace + 1.0, 1e-20))
            return np.stack(
                [
                    0.5 * root,
                    (m[:, 2, 1] - m[:, 1, 2]) / (2.0 * root),
                    (m[:, 0, 2] - m[:, 2, 0]) / (2.0 * root),
                    (m[:, 1, 0] - m[:, 0, 1]) / (2.0 * root),
                ],
                axis=1,
            )
        axis = index - 1
        other = [(axis + 1) % 3, (axis + 2) % 3]
        root = np.sqrt(
            np.maximum(1.0 + diagonal[:, axis] - diagonal[:, other[0]] - diagonal[:, other[1]], 1e-20)
        )
        parts = [np.empty_like(root) for _ in range(4)]
        parts[0] = (m[:, other[1], other[0]] - m[:, other[0], other[1]]) / (2.0 * root)
        parts[1 + axis] = 0.5 * root
        parts[1 + other[0]] = (m[:, other[0], axis] + m[:, axis, other[0]]) / (2.0 * root)
        parts[1 + other[1]] = (m[:, axis, other[1]] + m[:, other[1], axis]) / (2.0 * root)
        return np.stack(parts, axis=1)

    stacked = np.stack([branch(i) for i in range(4)], axis=0)
    chosen = stacked[case, np.arange(len(m))]
    # q and -q are the same rotation; pick w >= 0 so a block is comparable.
    return np.where(chosen[:, :1] < 0.0, -chosen, chosen)


def matrices_to_rotation_6d(matrices: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` -> ``(N, 6)``: the first two columns, as lerobot stores them."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"expected shape (N, 3, 3), got {matrices.shape}")
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)


def multiply_quaternions(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product ``left * right``, both ``(N, 4)`` ``[w, x, y, z]``.

    Composes rotations: the result applies ``right`` and then ``left``.
    """
    left = _as_quaternion_block(left)
    right = _as_quaternion_block(right)
    if len(left) != len(right) and 1 not in (len(left), len(right)):
        raise ValueError(f"cannot multiply {len(left)} by {len(right)} quaternions")
    w1, x1, y1, z1 = left.T
    w2, x2, y2, z2 = right.T
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def slerp_quaternions(start: np.ndarray, end: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Interpolate along the shorter arc, element-wise over ``(N, 4)`` blocks.

    ``fraction`` is ``(N,)`` in ``[0, 1]``: 0 gives ``start``, 1 gives ``end``.
    Rotations resampled onto a fixed grid belong on the arc between the two
    recorded samples, not at whichever is nearer -- at 20 Hz off a stalling
    recorder, nearest costs up to half a tick of angular error.

    Near-parallel pairs fall back to a normalised straight line, where the arc
    and the chord differ by less than the float noise but ``sin(theta)`` does not
    divide safely.
    """
    start = normalize_quaternions(_as_quaternion_block(start))
    end = normalize_quaternions(_as_quaternion_block(end))
    fraction = np.asarray(fraction, dtype=np.float64).reshape(-1, 1)

    dot = np.sum(start * end, axis=-1, keepdims=True)
    # q and -q are the same rotation, so flip to take the shorter arc.
    end = np.where(dot < 0.0, -end, end)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    parallel = dot > 0.9995
    safe_sin = np.where(parallel, 1.0, sin_theta)
    weight_start = np.where(parallel, 1.0 - fraction, np.sin((1.0 - fraction) * theta) / safe_sin)
    weight_end = np.where(parallel, fraction, np.sin(fraction * theta) / safe_sin)
    return normalize_quaternions(weight_start * start + weight_end * end)


def poses_7d_to_9d(poses: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """``(N, 7)`` ``[x, y, z] + quaternion`` -> ``(N, 9)`` ``[x, y, z] + 6D``."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"expected shape (N, 7), got {poses.shape}")
    rotations = quaternions_to_rotation_6d(to_wxyz(poses[:, 3:], order))
    return np.concatenate([poses[:, :3], rotations], axis=1)


def poses_9d_to_7d(poses: np.ndarray, order: str = "wxyz") -> np.ndarray:
    """``(N, 9)`` ``[x, y, z] + 6D`` -> ``(N, 7)`` ``[x, y, z] + quaternion``."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 9:
        raise ValueError(f"expected shape (N, 9), got {poses.shape}")
    quaternions = rotation_6d_to_quaternions(poses[:, 3:])
    if order == "xyzw":
        quaternions = quaternions[:, [1, 2, 3, 0]]
    elif order != "wxyz":
        raise ValueError(f"quaternion order must be one of {QUAT_ORDERS}, got {order!r}")
    return np.concatenate([poses[:, :3], quaternions], axis=1)


def _as_quaternion_block(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if quaternions.ndim == 1:
        quaternions = quaternions[None, :]
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"expected shape (N, 4), got {quaternions.shape}")
    return quaternions
